"""Free-model rotation with a circuit breaker, image trimming and the failure rules."""
from __future__ import annotations

import time
from typing import Callable

# fields only OpenRouter understands: removed before a request goes to any other provider
OPENROUTER_ONLY = ("provider", "usage", "reasoning", "models", "transforms", "route", "plugins", "include_reasoning")
# statuses that mean "this free model is not usable right now, try the next one"
FREE_RETRY_STATUS = (404, 408, 429, 500, 502, 503, 504)
# statuses from a PAID model that mean nothing was generated, so one fallback attempt cannot bill twice
OUTAGE_STATUS = (402, 502, 503, 504)


class FreeRotation:
    """Round-robin over free models. A model that fails is paused (default 10 minutes) and skipped until it recovers."""

    def __init__(self, pause_seconds: int = 600, clock: Callable[[], float] = time.time):
        self.pause_seconds = pause_seconds
        self.clock = clock
        self.paused: dict[str, float] = {}
        self._i = 0

    def pause(self, model: str) -> None:
        self.paused[model] = self.clock() + self.pause_seconds

    def active(self, models: list[str]) -> list[str]:
        now = self.clock()
        act = [m for m in models if self.paused.get(m, 0) <= now]
        return act or list(models)    # better to try a paused one than to fail for sure

    def next(self, models: list[str], exclude=()) -> str:
        cand = [m for m in self.active(models) if m not in exclude]
        if not cand:
            return ""
        pick = cand[self._i % len(cand)]
        self._i += 1
        return pick

    def retry_after_failure(self, models: list[str], current: str, exclude=()) -> str:
        """The next model in list order (circular) that has not been tried yet and is not paused."""
        if not models:
            return ""
        act = self.active([m for m in models if m not in exclude])
        base = models.index(current) if current in models else -1
        for k in range(1, len(models) + 1):
            cand = models[(base + k) % len(models)]
            if cand not in exclude and cand in act:
                self._i = models.index(cand) + 1
                return cand
        return ""

    def paused_now(self) -> dict[str, int]:
        now = self.clock()
        return {m: int(t - now) for m, t in self.paused.items() if t > now}


def free_failed(model: str, status: int, external: bool) -> bool:
    """status 0 = no connection."""
    return (status == 0 or status in FREE_RETRY_STATUS or (status == 403 and external)
            or (status == 402 and not external))   # out of credit: the :free models fail too


def strip_old_images(body: dict) -> bool:
    """Images only count in the LAST user message. Older ones are replaced by text, because clients resend the whole
    history on every call and an image would otherwise be billed again each time. True if the last user message has one."""
    items = body.get("input") if isinstance(body.get("input"), list) else body.get("messages")
    if not isinstance(items, list):
        return False
    is_img = lambda p: isinstance(p, dict) and p.get("type") in ("input_image", "image_url")
    last_user = max((i for i, m in enumerate(items) if isinstance(m, dict) and m.get("role") == "user"), default=-1)
    last_has = False
    for i, m in enumerate(items):
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list) or not any(is_img(p) for p in c):
            continue
        if i == last_user:
            last_has = True
            continue
        kind = "input_text" if body.get("input") is items else "text"
        m["content"] = [{"type": kind, "text": "[earlier image omitted]"} if is_img(p) else p for p in c]
    return last_has
