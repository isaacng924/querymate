"""QueryMate — a multi-agent NL→SQL analytics copilot (Phase 0 scaffold).

Phase 0 implements only the core agent loop: SQL Writer → Validator/Executor →
Critic (repair) → loop, plus an execution-accuracy eval harness. Schema-RAG,
planner, clarifier, and explainer arrive in later phases (see the vault spec
"QueryMate — NL-to-SQL Build Spec").
"""

__version__ = "0.0.1"
