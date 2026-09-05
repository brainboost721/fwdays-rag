import argparse

from rag.config import DATA_DIR, EMBED_BATCH, PROJECT_ROOT
from rag.embeddings import embed_many
from rag.loaders import iter_corpus_files, process_csv, process_md, process_pdf
from rag.store import get_collection

PROCESSORS = {".md": process_md, ".csv": process_csv, ".pdf": process_pdf}


def upsert_batches(collection, ids, documents, metadatas, batch_size=EMBED_BATCH):
    """Stable ids make upsert idempotent: re-indexing overwrites instead of duplicating."""
    for start in range(0, len(ids), batch_size):
        sl = slice(start, start + batch_size)
        collection.upsert(
            ids=ids[sl],
            documents=documents[sl],
            embeddings=embed_many(documents[sl]),
            metadatas=metadatas[sl],
        )


def build_index(rebuild: bool = False):
    collection = get_collection(rebuild=rebuild)
    files = iter_corpus_files(DATA_DIR)
    print(f"Знайдено файлів у {DATA_DIR.relative_to(PROJECT_ROOT)}: {len(files)}\n")

    for path in files:
        ids, docs, metas = PROCESSORS[path.suffix.lower()](path)
        if not docs:
            print(f"  пропуск (порожній): {path.relative_to(PROJECT_ROOT)}")
            continue
        upsert_batches(collection, ids, docs, metas)
        print(f"  {path.relative_to(PROJECT_ROOT)} → {len(docs)} чанків")

    print(f"\nГотово. Чанків у колекції: {collection.count()}")
    return collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Індексація корпусу data/ у ChromaDB")
    parser.add_argument("--rebuild", action="store_true", help="видалити колекцію і зібрати заново")
    build_index(rebuild=parser.parse_args().rebuild)


if __name__ == "__main__":
    main()
