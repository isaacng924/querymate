from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime config. Override any field via env (prefix ``QUERYMATE_``) or .env."""

    model_config = SettingsConfigDict(
        env_prefix="QUERYMATE_", env_file=".env", extra="ignore"
    )

    # --- Model routing -----------------------------------------------------
    # Per the build spec, QueryMate routes models for cost: a fast model writes
    # and repairs; the loop escalates to Opus on the final attempt. These are a
    # deliberate, spec-driven cost choice — set all three to "claude-opus-4-8"
    # if you'd rather pay for max quality everywhere.
    writer_model: str = "claude-sonnet-4-6"
    escalate_model: str = "claude-opus-4-8"  # used on the last allowed attempt
    fast_model: str = "claude-haiku-4-5"      # simple lookups (router tier 1)
    planner_model: str = "claude-haiku-4-5"   # advisory plan call

    # --- Retrieval (Phase 1) ---------------------------------------------
    retrieval_k: int = 5            # top-k schema cards per question
    schema_index_path: str = "data/schema_index.sqlite"

    # --- Critic loop -------------------------------------------------------
    max_attempts: int = 3  # repair attempts after the first write

    # --- Executor safety ---------------------------------------------------
    max_rows: int = 5000  # hard fetch cap (defence in depth, always enforced)
    statement_timeout_s: float = 10.0
    # Inject a LIMIT when the model omits one. ON for the interactive CLI (UX +
    # safety); the eval turns it OFF so a large gold result set isn't truncated.
    auto_limit: bool = True

    # --- Demo database -----------------------------------------------------
    demo_db_path: str = "data/demo_store.sqlite"

    # --- Observability (Phase 3) ------------------------------------------
    langfuse_enabled: bool = False  # no-op until wired in querymate/trace.py


# USD per MTok (input, output). Cache reads are billed at 0.1x input; cache
# writes at 1.25x input. Update when Anthropic prices change.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (15.0, 75.0),
}

settings = Settings()
