import os
from functools import lru_cache

from openai import OpenAI

from rag.config import EMBED_MODEL


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Немає OPENAI_API_KEY — додайте його у .env (зразок у .env.example)")
    return OpenAI()


def embed_many(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    resp = get_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]
