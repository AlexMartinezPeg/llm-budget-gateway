"""llm-budget-gateway: an OpenAI-compatible proxy with tier aliases, price ceilings, a daily budget and failover."""
from .app import create_app
from .config import Settings

__all__ = ["create_app", "Settings"]
__version__ = "0.1.0"
