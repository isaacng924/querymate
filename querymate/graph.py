"""Assemble the Phase-0 agent loop as a LangGraph state machine.

    START → write_sql → execute ─(ok / give_up)→ END
                           │
                        (error, attempts < max)
                           ▼
                        critic → write_sql  (loop)
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import critic_node, execute_node, route_after_execute, write_sql_node
from .state import QueryState

_GRAPH = None


def build_graph():
    g = StateGraph(QueryState)
    g.add_node("write_sql", write_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("critic", critic_node)

    g.add_edge(START, "write_sql")
    g.add_edge("write_sql", "execute")
    g.add_conditional_edges(
        "execute",
        route_after_execute,
        {"ok": END, "give_up": END, "critic": "critic"},
    )
    g.add_edge("critic", "write_sql")
    return g.compile()


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH
