"""Test fixtures: a fake upstream (no network), a controllable clock and a ready-to-use app."""
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_budget_gateway import Settings, create_app  # noqa: E402
from llm_budget_gateway.ledger import Ledger  # noqa: E402

T0 = 1_800_000_000.0   # a fixed moment (mid-day UTC) so the "day" does not change under the tests

MODELS = [
    # id, in, out (USD per token), input modalities, context
    ("p/cheap", 0.5e-6, 1e-6, ["text"], 64000),
    ("p/second", 0.5e-6, 1e-6, ["text"], 64000),
    ("p/expensive", 10e-6, 20e-6, ["text"], 64000),
    ("p/router", -1.0, -1.0, ["text"], 64000),
    ("v/vision", 1e-6, 2e-6, ["text", "image"], 64000),
    ("f/free1:free", 0.0, 0.0, ["text"], 8000),
    ("f/free2:free", 0.0, 0.0, ["text"], 16000),
    ("f/free3:free", 0.0, 0.0, ["text"], 4000),
]


def reply(status, obj):
    """A response that streams like a real network one (httpx.Response(json=...) comes pre-read and cannot be streamed)."""
    return httpx.Response(status, headers={"content-type": "application/json"}, stream=httpx.ByteStream(json.dumps(obj).encode()))


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


class Upstream:
    """Fake provider. `behavior` maps a model id or a host to an HTTP status (int), 'connect', 'readtimeout' or a callable."""

    def __init__(self):
        self.calls = []        # (host, path, body)
        self.behavior = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.calls.append((request.url.host, request.url.path, body))
        if request.url.path.endswith("/models"):
            if "list" in self.behavior:
                return self._act(self.behavior["list"], request, body)
            return reply(200, {"data": [
                {"id": i, "pricing": {"prompt": str(a), "completion": str(b)}, "context_length": c,
                 "architecture": {"input_modalities": mods, "output_modalities": ["text"]}} for i, a, b, mods, c in MODELS]})
        model = body.get("model", "")
        for key in (model, request.url.host):
            if key in self.behavior:
                return self._act(self.behavior[key], request, body)
        return self.ok(model)

    def _act(self, what, request, body):
        if callable(what):
            return what(request, body)
        if what == "connect":
            raise httpx.ConnectError("boom", request=request)
        if what == "readtimeout":
            raise httpx.ReadTimeout("slow", request=request)
        return reply(int(what), {"error": {"message": "scripted"}})

    @staticmethod
    def ok(model, cost=0.01, prompt=100, cached=40, served=None):
        return reply(200, {"id": "x", "model": served or model, "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                           "usage": {"prompt_tokens": prompt, "completion_tokens": 20, "cost": cost,
                                     "prompt_tokens_details": {"cached_tokens": cached}}})

    def chat_calls(self):
        return [c for c in self.calls if c[1].endswith("/chat/completions")]


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def upstream():
    return Upstream()


@pytest.fixture
def config_path(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "tiers": {"cheap": ["p/cheap", "p/second"], "vision": ["v/vision"], "big": ["p/expensive"], "mixed": ["p/cheap", "f/free1:free"]},
        "free_models": ["f/free1:free", "f/free2:free", "f/free3:free"],
        "outage_fallback": {"base_url": "https://fallback.example/v1", "api_key_env": "FB_KEY", "models": ["fb/model"]},
    }))
    return p


@pytest.fixture
def settings(tmp_path, clock, config_path):
    os.environ["FB_KEY"] = "fb-secret"
    return Settings(openrouter_key="or-secret", client_tokens=("tok-client",), admin_token="tok-admin", daily_budget_usd=1.0,
                    budget_ceiling_usd=5.0, data_dir=tmp_path / "data", config_path=config_path, clock=clock)


@pytest.fixture
def events():
    return []


@pytest.fixture
def client(settings, upstream, events):
    app = create_app(settings, transport=httpx.MockTransport(upstream), on_event=lambda k, m: events.append(k))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def ledger(settings, clock):
    return Ledger(settings.data_dir, settings.timezone, clock)


def spend(ledger, usd, model="p/cheap"):
    asyncio.run(ledger.record(model, usd, "seed"))


AUTH = {"Authorization": "Bearer tok-client"}


def chat(client, model="p/cheap", headers=None, **extra):
    body = {"model": model, "messages": [{"role": "user", "content": "hello"}], **extra}
    return client.post("/v1/chat/completions", json=body, headers={**AUTH, **(headers or {})})
