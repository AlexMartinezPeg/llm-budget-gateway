"""The gateway: an OpenAI-compatible proxy with tier aliases, a price guard, a daily budget, free-model rotation and
a one-shot failover when the upstream is down."""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
from typing import Callable

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import HotConfig, Settings
from .ledger import Ledger
from .pricing import PriceBook, cost_of, price_problem, served_model_and_tokens
from .routing import OPENROUTER_ONLY, OUTAGE_STATUS, FreeRotation, free_failed, strip_old_images

log = logging.getLogger("llm_budget_gateway")
ALLOWED_PATHS = {"chat/completions", "responses", "models", "embeddings"}


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None,
               on_event: Callable[[str, str], None] | None = None) -> FastAPI:
    """`transport` lets tests (or a proxy setup) replace the network; `on_event(kind, message)` is called at most once
    a day per kind for the things worth an alert (budget reached, free rotation started, upstream failover)."""
    s = settings or Settings.from_env()
    hot = HotConfig(s.config_path)
    ledger = Ledger(s.data_dir, s.timezone, s.clock)
    rotation = FreeRotation(s.free_pause_seconds, s.clock)
    new_client = lambda **kw: httpx.AsyncClient(transport=transport, **kw)
    book = PriceBook(s.upstream_base, lambda: new_client(), s.clock)
    sent: dict[str, str] = {}

    def event(kind: str, msg: str) -> None:
        if sent.get(kind) == ledger.today():
            return
        sent[kind] = ledger.today()
        log.warning("%s: %s", kind, msg)
        if on_event:
            try:
                on_event(kind, msg)
            except Exception:    # an alert hook must never break a request
                log.exception("on_event hook failed")

    # ---------------- auth ----------------
    def token_of(request: Request) -> str:
        tok = (request.headers.get("x-gateway-token") or "").strip()
        if not tok:
            a = request.headers.get("authorization") or ""
            if a.lower().startswith("bearer "):
                tok = a[7:].strip()
        return tok

    def matches(tok: str, valid) -> bool:
        return bool(tok) and any(hmac.compare_digest(tok.encode(), v.encode()) for v in valid if v)

    def client_auth(request: Request) -> None:
        if not matches(token_of(request), s.client_tokens):
            raise HTTPException(status_code=401, detail="Unauthorized")

    def admin_auth(request: Request) -> None:    # a client token must NOT be able to raise its own limit
        if not matches(token_of(request), (s.admin_token,)):
            raise HTTPException(status_code=401, detail="Unauthorized")

    # ---------------- model helpers ----------------
    def external(m: str) -> bool:
        return m in hot.outage_models()

    def is_free(m: str, prices: dict) -> bool:
        p = prices.get(m)
        return external(m) or (bool(p) and p[0] == 0.0 and p[1] == 0.0)

    def is_free_id(m: str, prices: dict) -> bool:
        """Free by name or price: it must never be a fallback for a paid model (no data-collection guarantee)."""
        return external(m) or m.endswith(":free") or is_free(m, prices)

    def free_list(prices: dict) -> list[str]:
        pref = hot.get().get("free_models")
        ids = [str(x) for x in pref] if isinstance(pref, list) and pref else list(book.free_detected)
        return [i for i in ids if is_free(i, prices)]

    def prep_free(body: dict, model: str, route: str) -> None:
        if external(model):
            for k in OPENROUTER_ONLY:
                body.pop(k, None)
            for m in body.get("messages") or []:   # some providers reject an assistant message with null content
                if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content") is None:
                    m["content"] = ""
        else:
            body.pop("models", None)    # a free model must never fall back to a paid one
            pr = dict(body["provider"]) if isinstance(body.get("provider"), dict) else {}
            pr["allow_fallbacks"] = False
            if not hot.get().get("free_allow_data_collection"):
                pr["data_collection"] = "deny"
            body["provider"] = pr
            if route == "chat/completions":
                body["usage"] = {"include": True}

    def target(model: str, url: str, headers: dict) -> tuple[str, dict]:
        if external(model):
            o = hot.outage
            key = os.environ.get(str(o.get("api_key_env") or ""), "")
            return (str(o["base_url"]).rstrip("/") + "/chat/completions",
                    {"Authorization": "Bearer " + key, "Content-Type": "application/json", "Accept-Encoding": "identity"})
        return url, headers

    app = FastAPI(title="llm-budget-gateway", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/usage")
    async def usage(request: Request):
        client_auth(request)
        out = ledger.summary(s.daily_budget_usd, s.budget_ceiling_usd)
        out["free_models_paused_seconds"] = rotation.paused_now()
        return out

    @app.post("/admin/budget")
    async def set_budget(request: Request):
        """Raise (or reset) TODAY's budget. Admin token only; it reverts by itself at midnight."""
        admin_auth(request)
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON")
        usd = data.get("usd") if isinstance(data, dict) else None
        if usd in (None, "", 0):
            ledger.clear_override()
            return {"ok": True, "budget_usd": s.daily_budget_usd, "override": False}
        try:
            usd = round(float(usd), 2)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="usd must be a number")
        if not (0.10 <= usd <= s.budget_ceiling_usd):
            raise HTTPException(status_code=400, detail=f"the override must be between 0.10 and {s.budget_ceiling_usd:.2f} USD")
        ledger.set_override(usd)
        event("override", f"today's budget set to {usd:.2f} USD (normal {s.daily_budget_usd:.2f})")
        return {"ok": True, "budget_usd": usd, "override": True, "normal": s.daily_budget_usd}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        client_auth(request)
        if path not in ALLOWED_PATHS:
            raise HTTPException(status_code=404, detail="path not allowed on this gateway")
        if not s.openrouter_key:
            raise HTTPException(status_code=503, detail="gateway has no upstream API key")
        cfg = hot.get()
        headers = {"Authorization": "Bearer " + s.openrouter_key, "Content-Type": "application/json", "Accept-Encoding": "identity"}
        url = f"{s.upstream_base}/{path}"
        caller = re.sub(r"[^a-z0-9_-]", "", (request.headers.get("x-client") or "").strip().lower())[:32] or "other"
        model, prices, payload, free, has_image = "", {}, None, False, False
        if request.method == "POST":
            try:
                body = await request.json()
                assert isinstance(body, dict)
            except Exception:
                raise HTTPException(status_code=400, detail="invalid JSON")
            model = str(body.get("model", ""))
            prices = await book.get()
            is_chat = path in ("responses", "chat/completions")
            if is_chat:
                has_image = bool(strip_old_images(body))
            models = [model]
            if model.startswith(s.alias_prefix):                      # tier alias -> real ids
                name = model[len(s.alias_prefix):]
                if not is_chat or (name == "free" and path != "chat/completions"):
                    raise HTTPException(status_code=400, detail=f"alias {model} is not valid on {path}")
                if name == "free":
                    alt = rotation.next(free_list(prices))
                    if not alt:
                        return JSONResponse({"error": {"type": "no_free_model", "message": "No free model available (empty free_models list)."}}, status_code=503)
                    models = [alt]
                else:
                    models = hot.tier(name)
                    if not models:
                        known = ", ".join(sorted((cfg.get("tiers") or {}).keys()) + ["free"])
                        raise HTTPException(status_code=400, detail=f"unknown alias: {model} (tiers: {known})")
            if has_image and book.vision:                              # image in the last message: need a vision model
                if models[0] not in book.vision:
                    models = hot.tier(str(cfg.get("vision_tier", "vision"))) or models
                models = models[:1] + [m for m in models[1:] if m in book.vision]
            pref = cfg.get("provider")
            if isinstance(pref, dict) and pref and "provider" not in body and is_chat:
                body["provider"] = pref
            model = models[0]
            free = is_free(model, prices)
            budget = ledger.budget(s.daily_budget_usd)
            spent = ledger.spent_today()
            if spent >= budget and not free:            # free models never trigger the budget stop
                event("budget", f"daily budget reached: {spent:.4f} of {budget:.2f} USD")
                alt = ""
                if path == "chat/completions" and not has_image and hot.free_fallback:
                    alt = rotation.next(free_list(prices))
                if not alt:
                    return JSONResponse({"error": {"type": "budget_exceeded", "message":
                        f"Daily budget reached ({spent:.2f}/{budget:.2f} USD) and no free model is available. It resets at midnight."}}, status_code=429)
                event("free", f"budget spent, rotating to free models: {model} -> {alt}")
                models, free = [alt], True
            if not prices and path != "models" and any(not external(m) for m in models):
                # without the price list neither the ceilings nor "is it free" can be checked: fail closed
                raise HTTPException(status_code=503, detail="no price list from upstream yet; retry in a minute")
            if prices and path != "models":
                problems = [price_problem(m, prices, s.max_in_usd_per_m, s.max_out_usd_per_m, s.allow_unknown_models, external(m)) for m in models]
                if all(problems):
                    raise HTTPException(status_code=403, detail=problems[0])
                models = [m for m, x in zip(models, problems) if not x]
                if isinstance(body.get("models"), list):   # a fallback list sent by the client faces the same rules
                    body["models"] = [m for m in body["models"] if isinstance(m, str) and not m.startswith(s.alias_prefix)
                                      and not price_problem(m, prices, s.max_in_usd_per_m, s.max_out_usd_per_m, s.allow_unknown_models, external(m))
                                      and not is_free_id(m, prices)]
                    if not body["models"]:
                        body.pop("models")
            model = body["model"] = models[0]
            free = free or is_free(model, prices)
            if not free:                                # a free model as backup of a paid one would go out without data protection
                models = models[:1] + [m for m in models[1:] if not is_free_id(m, prices)]
            if len(models) > 1 and path == "chat/completions" and not free:
                body["models"] = models                 # native upstream fallback: if the first fails, the next one answers
            if free and is_chat:
                prep_free(body, model, path)
            if path == "chat/completions":
                body.setdefault("max_tokens", 4096)
                if not external(model):
                    body["usage"] = {"include": True}
            elif path == "responses":
                body.setdefault("max_output_tokens", 8192)
            payload = json.dumps(body).encode()

        resolved = model
        http = new_client(timeout=httpx.Timeout(300, connect=15))
        tried = [model]
        max_attempts = 3 if free else 1
        switchable = (request.method == "POST" and path == "chat/completions" and not free and not has_image
                      and hot.free_fallback and bool(hot.outage_models()))
        resp, failure = None, None
        while True:
            u, h = target(model, url, headers)
            try:
                resp = await http.send(http.build_request(request.method, u, headers=h, content=payload), stream=True)
                status, failure = resp.status_code, None
            except Exception as e:
                resp, status, failure = None, 0, e
            alt = ""
            if free:
                if failure is not None and type(failure).__name__ not in ("ConnectError", "ConnectTimeout"):
                    break    # read timeout or cut mid-answer: no retry, nothing more is spent
                if not free_failed(model, status, external(model)):
                    break
                rotation.pause(model)                   # circuit breaker: rest this model, try the next
                if len(tried) < max_attempts:
                    alt = rotation.retry_after_failure(free_list(prices), model, tried)
            elif switchable and (status in OUTAGE_STATUS or type(failure).__name__ in ("ConnectError", "ConnectTimeout")):
                # Upstream down or out of credit: nothing was generated, so ONE attempt on the outage provider cannot bill twice.
                # A read timeout is never retried: the paid model may still be generating.
                alt = rotation.next(hot.outage_models())
                if alt:
                    switchable, free, max_attempts = False, True, len(tried) + 1
                    event("outage", f"upstream failed ({status or type(failure).__name__}), switching {model} -> {alt}")
            if not alt:
                break
            if resp is not None:
                await resp.aclose()
            log.info("%s failed (%s), trying %s", model, status or type(failure).__name__, alt)
            tried.append(alt)
            model = body["model"] = alt
            prep_free(body, model, path)
            payload = json.dumps(body).encode()
        if resp is None:
            await http.aclose()
            raise HTTPException(status_code=502, detail="upstream not responding: " + type(failure).__name__)

        async def stream():
            tail = ""
            try:
                async for chunk in resp.aiter_raw():
                    tail = (tail + chunk.decode("utf-8", "ignore"))[-60000:]
                    yield chunk
            finally:
                await resp.aclose()
                await http.aclose()
                if request.method == "POST" and resp.status_code < 400:
                    try:
                        served, tok_p, tok_c = served_model_and_tokens(tail)   # the model that actually answered
                        priced = served if served in prices else model
                        await ledger.record(served or model, cost_of(tail, priced, prices), caller, tok_p, tok_c)
                    except Exception as e:
                        log.warning("could not book the spend: %s", type(e).__name__)

        extra = {"X-Resolved-Model": re.sub(r"[^\x21-\x7e]", "", resolved)[:150]} if resolved else None
        return StreamingResponse(stream(), status_code=resp.status_code,
                                 media_type=resp.headers.get("content-type", "application/json"), headers=extra)

    return app
