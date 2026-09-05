import chromadb
from chromadb.errors import NotFoundError

from rag.config import CHROMA_DIR, COLLECTION_NAME


def get_collection(rebuild: bool = False):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except NotFoundError:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
