import json
import os
import time

from conftest import AUTH, chat, reply, spend

from llm_budget_gateway.routing import FreeRotation, free_failed, strip_old_images


# ------------------------------------------------------------------ auth and paths
def test_no_token_is_rejected(client):
    assert client.post("/v1/chat/completions", json={"model": "p/cheap"}).status_code == 401


def test_wrong_token_is_rejected(client):
    assert client.post("/v1/chat/completions", json={"model": "p/cheap"}, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_header_token_also_works(client):
    r = client.post("/v1/chat/completions", json={"model": "p/cheap", "messages": []}, headers={"X-Gateway-Token": "tok-client"})
    assert r.status_code == 200


def test_unlisted_path_is_404(client):
    assert client.get("/v1/files", headers=AUTH).status_code == 404


def test_invalid_json_is_400(client):
    r = client.post("/v1/chat/completions", content=b"{not json", headers=AUTH)
    assert r.status_code == 400


def test_health_needs_no_token(client):
    assert client.get("/health").json() == {"ok": True}


def test_no_upstream_key_is_503(settings, upstream):
    import httpx
    from fastapi.testclient import TestClient
    from llm_budget_gateway import create_app
    settings.openrouter_key = ""
    with TestClient(create_app(settings, transport=httpx.MockTransport(upstream))) as c:
        assert chat(c).status_code == 503


# ------------------------------------------------------------------ tiers
def test_tier_alias_resolves_primary_and_sends_fallback_list(client, upstream):
    r = chat(client, "tier/cheap")
    assert r.status_code == 200
    sent = upstream.chat_calls()[0][2]
    assert sent["model"] == "p/cheap" and sent["models"] == ["p/cheap", "p/second"]


def test_unknown_tier_is_400_and_lists_the_known_ones(client):
    r = chat(client, "tier/nope")
    assert r.status_code == 400 and "cheap" in r.json()["detail"]


def test_real_model_ids_pass_through_unchanged(client, upstream):
    chat(client, "p/second")
    assert upstream.chat_calls()[0][2]["model"] == "p/second"


def test_resolved_model_header(client):
    assert chat(client, "tier/cheap").headers["X-Resolved-Model"] == "p/cheap"


def test_a_free_model_is_removed_from_the_backup_of_a_paid_model(client, upstream):
    chat(client, "tier/mixed")
    sent = upstream.chat_calls()[0][2]
    assert "models" not in sent and sent["model"] == "p/cheap"


def test_tiers_reload_without_restart(client, upstream, config_path):
    chat(client, "tier/cheap")
    cfg = json.loads(config_path.read_text())
    cfg["tiers"]["cheap"] = ["p/second"]
    config_path.write_text(json.dumps(cfg))
    later = time.time() + 10
    os.utime(config_path, (later, later))
    chat(client, "tier/cheap")
    assert upstream.chat_calls()[1][2]["model"] == "p/second"


def test_defaults_are_added(client, upstream):
    chat(client)
    sent = upstream.chat_calls()[0][2]
    assert sent["max_tokens"] == 4096 and sent["usage"] == {"include": True}


def test_client_max_tokens_is_respected(client, upstream):
    chat(client, max_tokens=50)
    assert upstream.chat_calls()[0][2]["max_tokens"] == 50


def test_provider_preferences_from_config_apply_only_when_the_client_sends_none(client, upstream, config_path):
    cfg = json.loads(config_path.read_text())
    cfg["provider"] = {"sort": "price"}
    config_path.write_text(json.dumps(cfg))
    later = time.time() + 10
    os.utime(config_path, (later, later))
    chat(client)
    chat(client, provider={"order": ["x"]})
    assert upstream.chat_calls()[0][2]["provider"] == {"sort": "price"}
    assert upstream.chat_calls()[1][2]["provider"] == {"order": ["x"]}


# ------------------------------------------------------------------ price guard
def test_expensive_model_is_403(client):
    r = chat(client, "p/expensive")
    assert r.status_code == 403 and "too expensive" in r.json()["detail"]


def test_unknown_model_is_403(client):
    assert chat(client, "who/knows").status_code == 403


def test_variable_price_router_is_403(client):
    assert "variable-price" in chat(client, "p/router").json()["detail"]


def test_expensive_tier_is_403(client):
    assert chat(client, "tier/big").status_code == 403


def test_expensive_backup_from_the_client_is_dropped(client, upstream):
    chat(client, "p/cheap", models=["p/expensive", "p/second", "tier/cheap"])
    assert upstream.chat_calls()[0][2]["models"] == ["p/second"]


def test_missing_price_list_fails_closed(client, upstream):
    upstream.behavior["list"] = 500
    assert chat(client).status_code == 503


# ------------------------------------------------------------------ budget and accounting
def test_cost_is_booked_to_the_model_that_served_and_to_the_caller(client, upstream, ledger):
    upstream.behavior["p/cheap"] = lambda req, body: upstream.ok("p/cheap", cost=0.25, served="p/second")
    chat(client, headers={"X-Client": "My App!"})
    u = client.get("/usage", headers=AUTH).json()
    assert u["spent_usd"] == 0.25 and u["by_model"] == {"p/second": 0.25}
    assert u["by_client"]["myapp"]["calls"] == 1     # label sanitised to [a-z0-9_-]


def test_cached_tokens_are_tracked(client):
    chat(client)
    u = client.get("/usage", headers=AUTH).json()
    assert u["cache"]["p/cheap"] == {"prompt": 100, "cached": 40, "hit_pct": 40.0}


def test_cost_falls_back_to_tokens_times_price(client, upstream):
    upstream.behavior["p/cheap"] = lambda req, body: reply(
        200, {"model": "p/cheap", "usage": {"prompt_tokens": 1000, "completion_tokens": 500}})
    chat(client)
    assert abs(client.get("/usage", headers=AUTH).json()["spent_usd"] - (1000 * 0.5e-6 + 500 * 1e-6)) < 1e-9


def test_budget_reached_reroutes_chat_to_a_free_model(client, upstream, ledger, events):
    spend(ledger, 1.5)
    r = chat(client, "p/cheap")
    assert r.status_code == 200
    sent = upstream.chat_calls()[0][2]
    assert sent["model"] == "f/free1:free"
    assert sent["provider"] == {"allow_fallbacks": False, "data_collection": "deny"}
    assert "models" not in sent
    assert "budget" in events and "free" in events


def test_free_models_do_not_trigger_the_budget_stop(client, ledger):
    spend(ledger, 5.0)
    assert chat(client, "f/free2:free").status_code == 200


def test_budget_reached_and_no_free_models_is_429(client, ledger, config_path):
    spend(ledger, 1.5)
    cfg = json.loads(config_path.read_text())
    cfg["free_fallback"] = False
    config_path.write_text(json.dumps(cfg))
    later = time.time() + 10
    os.utime(config_path, (later, later))
    r = chat(client)
    assert r.status_code == 429 and r.json()["error"]["type"] == "budget_exceeded"


def test_embeddings_are_not_rerouted_to_free_chat_models(client, ledger):
    spend(ledger, 1.5)
    r = client.post("/v1/embeddings", json={"model": "p/cheap", "input": "x"}, headers=AUTH)
    assert r.status_code == 429


def test_images_are_never_rerouted_to_free_models(client, ledger):
    spend(ledger, 1.5)
    msg = [{"role": "user", "content": [{"type": "text", "text": "see"}, {"type": "image_url", "image_url": {"url": "data:x"}}]}]
    assert client.post("/v1/chat/completions", json={"model": "v/vision", "messages": msg}, headers=AUTH).status_code == 429


def test_a_new_day_resets_the_budget(client, ledger, clock, upstream):
    spend(ledger, 1.5)
    clock.t += 86400
    chat(client, "p/cheap")
    assert upstream.chat_calls()[0][2]["model"] == "p/cheap"


def test_usage_requires_a_client_token(client):
    assert client.get("/usage").status_code == 401


# ------------------------------------------------------------------ admin override
def test_a_client_token_cannot_raise_the_budget(client):
    r = client.post("/admin/budget", json={"usd": 3}, headers=AUTH)
    assert r.status_code == 401


def test_admin_raises_todays_budget_and_it_reverts_tomorrow(client, ledger, clock, upstream):
    spend(ledger, 1.5)
    admin = {"Authorization": "Bearer tok-admin"}
    assert client.post("/admin/budget", json={"usd": 3}, headers=admin).json()["budget_usd"] == 3.0
    chat(client, "p/cheap")
    assert upstream.chat_calls()[0][2]["model"] == "p/cheap"          # not rerouted: 1.5 < 3
    clock.t += 86400
    assert client.get("/usage", headers=AUTH).json()["budget_usd"] == 1.0


def test_admin_override_is_limited_to_the_ceiling(client):
    admin = {"Authorization": "Bearer tok-admin"}
    assert client.post("/admin/budget", json={"usd": 50}, headers=admin).status_code == 400
    assert client.post("/admin/budget", json={"usd": 0.01}, headers=admin).status_code == 400
    assert client.post("/admin/budget", json={"usd": "abc"}, headers=admin).status_code == 400


def test_admin_can_reset_the_override(client):
    admin = {"Authorization": "Bearer tok-admin"}
    client.post("/admin/budget", json={"usd": 3}, headers=admin)
    assert client.post("/admin/budget", json={"usd": None}, headers=admin).json()["override"] is False
    assert client.get("/usage", headers=AUTH).json()["budget_usd"] == 1.0


# ------------------------------------------------------------------ free rotation and circuit breaker
def test_a_failing_free_model_is_paused_and_the_next_one_answers(client, upstream, ledger, clock):
    spend(ledger, 1.5)
    upstream.behavior["f/free1:free"] = 429
    r = chat(client)
    assert r.status_code == 200
    assert [c[2]["model"] for c in upstream.chat_calls()] == ["f/free1:free", "f/free2:free"]
    assert "f/free1:free" in client.get("/usage", headers=AUTH).json()["free_models_paused_seconds"]
    upstream.calls.clear()
    chat(client)
    assert "f/free1:free" not in [c[2]["model"] for c in upstream.chat_calls()]
    clock.t += 601
    assert client.get("/usage", headers=AUTH).json()["free_models_paused_seconds"] == {}


def test_free_requests_stop_after_three_attempts(client, upstream, ledger):
    spend(ledger, 1.5)
    for m in ("f/free1:free", "f/free2:free", "f/free3:free"):
        upstream.behavior[m] = 503
    r = chat(client)
    assert r.status_code == 503 and len(upstream.chat_calls()) == 3


def test_free_models_never_get_a_paid_fallback_list(client, upstream):
    chat(client, "f/free1:free", models=["p/cheap"])
    assert "models" not in upstream.chat_calls()[0][2]


def test_free_alias_picks_from_the_free_list(client, upstream):
    assert chat(client, "tier/free").status_code == 200
    assert upstream.chat_calls()[0][2]["model"].endswith(":free")


# ------------------------------------------------------------------ outage failover
def test_upstream_503_makes_exactly_one_attempt_on_the_outage_provider(client, upstream, events):
    upstream.behavior["openrouter.ai"] = 503
    r = chat(client, "p/cheap", provider={"sort": "price"}, reasoning={"effort": "low"})
    assert r.status_code == 200
    calls = upstream.chat_calls()
    assert [c[0] for c in calls] == ["openrouter.ai", "fallback.example"]
    sent = calls[1][2]
    assert sent["model"] == "fb/model" and "provider" not in sent and "reasoning" not in sent and "usage" not in sent
    assert "outage" in events


def test_connection_error_also_fails_over(client, upstream):
    upstream.behavior["openrouter.ai"] = "connect"
    assert chat(client).status_code == 200


def test_the_outage_attempt_is_not_repeated_when_it_fails_too(client, upstream):
    upstream.behavior["openrouter.ai"] = 503
    upstream.behavior["fallback.example"] = 503
    r = chat(client)
    assert r.status_code == 503 and len(upstream.chat_calls()) == 2


def test_a_read_timeout_is_never_retried(client, upstream):
    upstream.behavior["p/cheap"] = "readtimeout"
    r = chat(client)
    assert r.status_code == 502 and len(upstream.chat_calls()) == 1


def test_a_normal_400_is_returned_as_is_without_failover(client, upstream):
    upstream.behavior["p/cheap"] = 400
    assert chat(client).status_code == 400 and len(upstream.chat_calls()) == 1


def test_outage_provider_receives_assistant_messages_without_null_content(client, upstream):
    upstream.behavior["openrouter.ai"] = 503
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": None, "tool_calls": []}]
    client.post("/v1/chat/completions", json={"model": "p/cheap", "messages": msgs}, headers=AUTH)
    assert upstream.chat_calls()[1][2]["messages"][1]["content"] == ""


# ------------------------------------------------------------------ images
def test_vision_model_is_used_when_the_last_message_has_an_image(client, upstream):
    msg = [{"role": "user", "content": [{"type": "text", "text": "see"}, {"type": "image_url", "image_url": {"url": "data:x"}}]}]
    client.post("/v1/chat/completions", json={"model": "p/cheap", "messages": msg}, headers=AUTH)
    assert upstream.chat_calls()[0][2]["model"] == "v/vision"


def test_old_images_are_replaced_by_text():
    body = {"messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "old"}}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "now"}, {"type": "image_url", "image_url": {"url": "new"}}]}]}
    assert strip_old_images(body) is True
    assert body["messages"][0]["content"] == [{"type": "text", "text": "[earlier image omitted]"}]
    assert body["messages"][2]["content"][1]["image_url"]["url"] == "new"


def test_no_image_in_the_last_message_returns_false():
    body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}, {"role": "user", "content": "plain"}]}
    assert strip_old_images(body) is False


# ------------------------------------------------------------------ small units
def test_rotation_is_round_robin_and_skips_paused_models():
    t = [1000.0]
    r = FreeRotation(600, lambda: t[0])
    ms = ["a", "b", "c"]
    assert [r.next(ms) for _ in range(4)] == ["a", "b", "c", "a"]
    r.pause("b")
    assert "b" not in {r.next(ms) for _ in range(6)}
    t[0] += 601
    assert "b" in {r.next(ms) for _ in range(6)}


def test_if_every_free_model_is_paused_it_still_tries_one():
    r = FreeRotation(600, lambda: 0.0)
    for m in ("a", "b"):
        r.pause(m)
    assert r.next(["a", "b"]) in ("a", "b")


def test_free_failure_rules():
    assert free_failed("x:free", 429, False) and free_failed("x:free", 0, False)
    assert free_failed("x:free", 402, False) and not free_failed("fb/m", 402, True)
    assert free_failed("fb/m", 403, True) and not free_failed("x:free", 403, False)
    assert not free_failed("x:free", 200, False) and not free_failed("x:free", 400, False)
