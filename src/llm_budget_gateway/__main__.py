"""Run with:  python -m llm_budget_gateway   (settings come from environment variables, see the README)."""
import os

import uvicorn

from . import create_app

if __name__ == "__main__":
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
