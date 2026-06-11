"""Phase-3 observability seam (Langfuse). No-op for now.

When Langfuse is wired in, ``trace()`` becomes one span per question with child
spans per writer/critic call and per SQL execution (latency, tokens, cost,
model, retry path). Kept as a context-manager seam so the call sites already
exist.
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def trace(name: str, **meta):
    # TODO(Phase 3): start a Langfuse trace/span here when settings.langfuse_enabled.
    yield
