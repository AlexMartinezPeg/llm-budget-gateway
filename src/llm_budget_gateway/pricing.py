"""Model price book (from the provider's public model list) and the price guard."""
from __future__ import annotations

import logging
import re
import time
from typing import Callable

import httpx

log = logging.getLogger("llm_budget_gateway")
INF = float("inf")


class PriceBook:
    """Caches {model_id: (usd_per_input_token, usd_per_output_token)} for an hour. After a failed refresh it retries at
    most once a minute, and keeps serving the last good list."""

    def __init__(self, base_url: str, client_factory: Callable[[], httpx.AsyncClient], clock: Callable[[], float] = time.time):
        self.base = base_url
        self.client_factory = client_factory
        self.clock = clock
        self.prices: dict[str, tuple[float, float]] = {}
        self.vision: set[str] = set()
        self.free_detected: list[str] = []     # free text models, longest context first
        self._ts = 0.0
        self._failed_at = 0.0

    async def get(self) -> dict[str, tuple[float, float]]:
        now = self.clock()
        if self.prices and now - self._ts < 3600:
            return self.prices
        if now - self._failed_at < 60:
            return self.prices
        try:
            async with self.client_factory() as hc:
                r = await hc.get(f"{self.base}/models", timeout=20)
            data = r.json().get("data", [])
            prices = {x["id"]: (float(x["pricing"]["prompt"]), float(x["pricing"]["completion"])) for x in data if x.get("pricing")}
            vision = {x["id"] for x in data if "image" in ((x.get("architecture") or {}).get("input_modalities") or [])}
            if vision:
                self.vision = vision
            skip = ("safety", "guard", "embed", "whisper", "tts")
            free = [(x["id"], int(x.get("context_length") or 0)) for x in data
                    if prices.get(x.get("id")) == (0.0, 0.0)
                    and ((x.get("architecture") or {}).get("output_modalities") or ["text"]) == ["text"]
                    and not any(k in x["id"] for k in skip)]
            self.free_detected = [i for i, _ in sorted(free, key=lambda t: -t[1])]
            if prices:
                self.prices, self._ts = prices, now
            else:
                self._failed_at = now
        except Exception as e:   # network, JSON, schema: keep the old list
            self._failed_at = now
            log.warning("price list refresh failed: %s", type(e).__name__)
        return self.prices


def price_problem(model: str, prices: dict, max_in_per_m: float, max_out_per_m: float, allow_unknown: bool, external: bool = False) -> str:
    """'' if the model passes the price guard, otherwise the reason it is refused (the gateway answers 403)."""
    if not prices or external:
        return ""
    p = prices.get(model) or prices.get(model.split(":")[0])
    if p is None:
        return "" if allow_unknown else f"unknown or not allowed model: {model}"
    try:
        pin, pout = float(p[0]), float(p[1])
    except (TypeError, ValueError):
        return f"model not allowed (invalid price): {model}"
    if not (0.0 <= pin < INF and 0.0 <= pout < INF):
        # routers such as "auto" publish -1/-1 (variable price) and would slip past any ceiling
        return f"model not allowed (variable-price router): {model}"
    if pin * 1e6 > max_in_per_m or pout * 1e6 > max_out_per_m:
        return f"model too expensive for this gateway: {model}"
    return ""


# ---- reading usage out of the (possibly streamed) response tail ----
def served_model_and_tokens(text: str) -> tuple[str, int, int]:
    """(model that actually answered, prompt tokens, cached prompt tokens) from the last blocks of the response."""
    ms = re.findall(r'"model"\s*:\s*"([A-Za-z0-9/._:@-]{1,120})"', text)
    tp = re.findall(r'"(?:prompt_tokens|input_tokens)"\s*:\s*(\d+)', text)
    tc = re.findall(r'"cached_tokens"\s*:\s*(\d+)', text)
    return (ms[-1] if ms else "", int(tp[-1]) if tp else 0, int(tc[-1]) if tc else 0)


def cost_of(text: str, model: str, prices: dict) -> float:
    """The provider-reported cost if present, else tokens x listed price."""
    ms = re.findall(r'"cost"\s*:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', text)
    if ms:
        return float(ms[-1])
    tin = re.findall(r'"(?:prompt_tokens|input_tokens)"\s*:\s*(\d+)', text)
    tout = re.findall(r'"(?:completion_tokens|output_tokens)"\s*:\s*(\d+)', text)
    pin, pout = prices.get(model, prices.get(model.split(":")[0], (0.0, 0.0)))
    return (int(tin[-1]) if tin else 0) * pin + (int(tout[-1]) if tout else 0) * pout
