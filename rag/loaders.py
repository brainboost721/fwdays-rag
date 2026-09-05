import csv
import re
from pathlib import Path

import pymupdf

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, PROJECT_ROOT

SUPPORTED_SUFFIXES = {".md", ".csv", ".pdf"}

Records = tuple[list[str], list[str], list[dict]]


def iter_corpus_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def chunker(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _id_base(rel_path: str) -> str:
    return rel_path.replace("/", "__").replace("\\", "__")


def _records(parts: list[str], id_prefix: str, source: str) -> Records:
    ids = [f"{id_prefix}:part{i:04d}" for i in range(len(parts))]
    metas = [{"source": source} for _ in parts]
    return ids, parts, metas


def process_md(path: Path) -> Records:
    rel = _rel(path)
    text = path.read_text(encoding="utf-8").strip()
    return _records(chunker(text), _id_base(rel), rel)


def process_pdf(path: Path) -> Records:
    rel = _rel(path)
    doc = pymupdf.open(path)
    try:
        text = "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return _records(chunker(text), _id_base(rel), rel)


def process_csv(path: Path) -> Records:
    """One CSV row is one logical document."""
    rel = _rel(path)
    base = _id_base(rel)
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    with path.open(encoding="utf-8", newline="") as f:
        for row_idx, row in enumerate(csv.DictReader(f)):
            row_key = (row.get("ticket_id") or row.get("service_id") or f"row{row_idx}").strip()
            fields = [f"- **{k}**: {v}" for k, v in row.items() if v and v.strip()]

            # Repeat the header in every chunk so a split row keeps its record id.
            header = f"# {path.name} — рядок {row_idx + 1} · {row_key}"
            parts = [f"{header}\n\n{part}" for part in chunker("\n".join(fields))]

            prefix = f"{base}__row{row_idx:04d}_{re.sub(r'[^\w\-]+', '_', row_key)}"
            row_ids, row_docs, row_metas = _records(parts, prefix, rel)
            ids += row_ids
            docs += row_docs
            metas += row_metas

    return ids, docs, metas
