from rag.config import MAX_DISTANCE, TOP_K
from rag.embeddings import embed_many
from rag.store import get_collection

Hit = dict


def retrieve(
    query: str,
    k: int = TOP_K,
    where: dict | None = None,
    max_distance: float = MAX_DISTANCE,
) -> list[Hit]:
    """Top-k nearest chunks, with weakly related ones dropped by the distance cutoff."""
    res = get_collection().query(
        query_embeddings=embed_many([query]),
        n_results=k,
        where=where,
        include=["documents", "distances", "metadatas"],
    )
    return [
        {"document": doc, "source": meta["source"], "distance": dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
        if dist <= max_distance
    ]
