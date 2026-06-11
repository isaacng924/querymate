"""Schema-card retrieval: KNN top-k per db_id, then FK 1-hop expansion so join
paths survive, then the writer's schema string built from the selected cards."""

from __future__ import annotations

from .card_index import CardIndex
from .schema_cards import SchemaCard


class Retriever:
    def __init__(self, index: CardIndex) -> None:
        self.index = index

    def retrieve(
        self, question: str, *, db_id: str, k: int
    ) -> tuple[list[SchemaCard], str]:
        hits = self.index.query(question, db_id=db_id, k=k)
        seen = {c.table for c in hits}
        neighbor_tables = [
            t for c in hits for t in c.fk_neighbors if t not in seen and not seen.add(t)
        ]
        cards = hits + self.index.get_cards(db_id, neighbor_tables)
        schema = "\n\n".join(c.text for c in cards)
        return cards, schema
