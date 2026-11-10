"""Settings (environment) and the hot-reloaded JSON config (tiers, free models, provider preferences)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass
class Settings:
    openrouter_key: str = ""
    client_tokens: tuple[str, ...] = ()      # tokens your tools use to call the gateway
    admin_token: str = ""                    # only this one may raise today's budget
    daily_budget_usd: float = 1.0
    budget_ceiling_usd: float = 5.0          # the most an admin override may set
    max_in_usd_per_m: float = 3.0            # price ceilings, USD per million tokens
    max_out_usd_per_m: float = 6.0
    allow_unknown_models: bool = False
    data_dir: Path = Path("./data")
    config_path: Path | None = None
    upstream_base: str = "https://openrouter.ai/api/v1"
    timezone: str = "UTC"                    # the day (and so the budget) rolls over at midnight in this zone
    alias_prefix: str = "tier/"              # "tier/cheap" -> list of real model ids from the config
    free_pause_seconds: int = 600            # a free model that fails rests this long (circuit breaker)
    clock: Callable[[], float] = field(default=time.time, repr=False)

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        tokens = tuple(t.strip() for t in (e("GATEWAY_TOKENS", "")).split(",") if t.strip())
        cfg = e("GATEWAY_CONFIG")
        return cls(
            openrouter_key=e("OPENROUTER_API_KEY", ""),
            client_tokens=tokens,
            admin_token=e("ADMIN_TOKEN", ""),
            daily_budget_usd=_float("DAILY_BUDGET_USD", 1.0),
            budget_ceiling_usd=_float("BUDGET_CEILING_USD", 5.0),
            max_in_usd_per_m=_float("MAX_IN_USD_PER_M", 3.0),
            max_out_usd_per_m=_float("MAX_OUT_USD_PER_M", 6.0),
            allow_unknown_models=bool(e("ALLOW_UNKNOWN_MODELS")),
            data_dir=Path(e("DATA_DIR", "./data")),
            config_path=Path(cfg) if cfg else None,
            upstream_base=e("UPSTREAM_BASE", "https://openrouter.ai/api/v1").rstrip("/"),
            timezone=e("GATEWAY_TZ", "UTC"),
        )


class HotConfig:
    """The JSON config is re-read whenever the file changes, so tiers and free models can be edited without a restart.

    Keys (all optional):
      tiers:               {"cheap": ["primary-id", "fallback-id"], ...}
      free_models:         ["id", ...] free models used when the budget runs out
      free_fallback:       false turns the free-model rotation off
      free_allow_data_collection: true lets free models log prompts (default: denied)
      provider:            OpenRouter provider preferences added to requests that do not send their own
      vision_tier:         tier used when the last message carries an image (default "vision")
      outage_fallback:     {"base_url": "...", "api_key_env": "NAME", "models": ["id"]}  one attempt when upstream is down
    """

    def __init__(self, path: Path | None):
        self.path = path
        self._mtime = 0.0
        self._data: dict[str, Any] = {}

    def get(self) -> dict[str, Any]:
        if not self.path:
            return self._data
        try:
            mtime = os.path.getmtime(self.path)
            if mtime != self._mtime:
                with open(self.path, encoding="utf-8") as fh:
                    data = json.load(fh)
                self._data = data if isinstance(data, dict) else {}
                self._mtime = mtime
        except (OSError, ValueError):
            pass   # keep the last good config
        return self._data

    def tier(self, name: str) -> list[str] | None:
        t = self.get().get("tiers")
        if isinstance(t, dict):
            v = t.get(name)
            v = [v] if isinstance(v, str) and v else v
            if isinstance(v, list) and v and all(isinstance(x, str) and x for x in v):
                return list(v)
        return None

    @property
    def free_fallback(self) -> bool:
        return self.get().get("free_fallback", True) is not False

    @property
    def outage(self) -> dict[str, Any]:
        o = self.get().get("outage_fallback")
        return o if isinstance(o, dict) and o.get("base_url") and isinstance(o.get("models"), list) else {}

    def outage_models(self) -> list[str]:
        return [m for m in self.outage.get("models", []) if isinstance(m, str)]
