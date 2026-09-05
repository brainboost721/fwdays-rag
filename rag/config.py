import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_data"
COLLECTION_NAME = "yaic_support"

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

CHUNK_SIZE = 420
CHUNK_OVERLAP = 90
EMBED_BATCH = 64

TOP_K = 6

# Backstop for far-off queries only (other language, gibberish): those measure 0.78+,
# while every in-corpus question stays under 0.51. Questions that are merely absent from
# the corpus land at 0.52-0.73 and pass this cutoff on purpose — refusing them is the
# prompt's job, not the retriever's.
MAX_DISTANCE = 0.76
