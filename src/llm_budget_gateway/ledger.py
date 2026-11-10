"""Daily spend ledger: per day, model, hour and calling client, plus provider-cache hit tokens.

The budget is a SOFT cap: it is checked before a call and booked after it, so a burst of parallel calls or a very
large max_tokens can overshoot it a little. Atomic writes (temp file + replace) keep the file valid.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo


class Ledger:
    def __init__(self, data_dir: Path, tz: str = "UTC", clock: Callable[[], float] = time.time):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "ledger.json"
        self.override_path = self.dir / "budget_override.json"
        self.tz = ZoneInfo(tz)
        self.clock = clock
        self._lock = asyncio.Lock()

    # ----- time -----
    def _now(self) -> datetime:
        return datetime.fromtimestamp(self.clock(), self.tz)

    def today(self) -> str:
        return self._now().strftime("%Y-%m-%d")

    def hour(self) -> str:
        return self._now().strftime("%H")

    # ----- storage -----
    def load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, d: dict) -> None:
        for k in sorted(d.keys())[:-30]:    # keep the last 30 days
            d.pop(k, None)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)
        os.replace(tmp, self.path)

    # ----- accounting -----
    async def record(self, model: str, cost: float, client: str = "other", prompt_tokens: int = 0, cached_tokens: int = 0) -> None:
        async with self._lock:
            d = self.load()
            day = d.setdefault(self.today(), {"usd": 0.0, "calls": 0, "models": {}})
            day["usd"] = round(day.get("usd", 0.0) + cost, 6)
            day["calls"] = day.get("calls", 0) + 1
            day.setdefault("models", {})[model] = round(day["models"].get(model, 0.0) + cost, 6)
            hours = day.setdefault("hours", {})
            hours[self.hour()] = round(hours.get(self.hour(), 0.0) + cost, 6)
            c = day.setdefault("clients", {}).setdefault(client, {"usd": 0.0, "calls": 0})
            c["usd"] = round(c.get("usd", 0.0) + cost, 6)
            c["calls"] = c.get("calls", 0) + 1
            if prompt_tokens:
                cache = day.setdefault("cache", {}).setdefault(model, {"prompt": 0, "cached": 0})
                cache["prompt"] += int(prompt_tokens)
                cache["cached"] += int(cached_tokens)
            self._save(d)

    def spent_today(self) -> float:
        return float(self.load().get(self.today(), {}).get("usd", 0.0))

    # ----- budget override (admin only, today only) -----
    def budget(self, base: float) -> float:
        try:
            with open(self.override_path, encoding="utf-8") as fh:
                o = json.load(fh)
            if o.get("date") == self.today():
                return float(o["usd"])
        except (OSError, ValueError, KeyError):
            pass
        return base

    def set_override(self, usd: float) -> None:
        with open(self.override_path, "w", encoding="utf-8") as fh:
            json.dump({"date": self.today(), "usd": usd}, fh)

    def clear_override(self) -> None:
        try:
            os.remove(self.override_path)
        except FileNotFoundError:
            pass

    def summary(self, base: float, ceiling: float) -> dict:
        d = self.load()
        today = d.get(self.today(), {"usd": 0.0, "calls": 0, "models": {}})
        budget = self.budget(base)
        cache = {m: {**v, "hit_pct": round(100.0 * v["cached"] / v["prompt"], 1) if v.get("prompt") else 0.0}
                 for m, v in today.get("cache", {}).items()}
        return {
            "date": self.today(), "spent_usd": today.get("usd", 0.0), "budget_usd": budget, "base_budget_usd": base,
            "override_active": budget != base, "budget_ceiling_usd": ceiling,
            "remaining_usd": round(max(0.0, budget - today.get("usd", 0.0)), 6), "calls": today.get("calls", 0),
            "by_model": today.get("models", {}), "by_client": today.get("clients", {}), "cache": cache,
            "last_days": [{"date": k, "usd": v.get("usd", 0.0), "calls": v.get("calls", 0)} for k, v in sorted(d.items())[-14:]],
        }
