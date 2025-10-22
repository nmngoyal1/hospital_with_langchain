# app/services/lc_vector.py
import os
from pathlib import Path
from typing import Any, Dict, List

# 🔇 silence Chroma telemetry & misc warnings BEFORE importing anything Chroma-ish
os.environ.setdefault("CHROMADB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain_chroma import Chroma  # ✅ new package
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ modern embeddings

PERSIST_DIR = Path("storage/chroma").as_posix()
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_BATCH = 100  # keep < 166 on Chroma 0.5.x

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def normalize_metadata_value(v: Any) -> Any:
    if isinstance(v, list):
        return "|".join(map(str, v))
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)

def normalize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {k: normalize_metadata_value(v) for k, v in meta.items()}

def build_vectorstore_from_documents(documents: List, persist_dir: str = PERSIST_DIR) -> Chroma:
    """
    Create/update a Chroma store and add documents in safe batches.
    Chroma >=0.4 auto-persists; no manual persist() call needed.
    """
    embeddings = get_embeddings()
    vs = Chroma(embedding_function=embeddings, persist_directory=persist_dir)

    # Add in batches to avoid "Batch size exceeds maximum" errors
    for i in range(0, len(documents), MAX_BATCH):
        vs.add_documents(documents[i:i + MAX_BATCH])

    # No vs.persist() needed (auto)
    return vs

def load_vectorstore(persist_dir: str = PERSIST_DIR) -> Chroma:
    embeddings = get_embeddings()
    return Chroma(embedding_function=embeddings, persist_directory=persist_dir)

def get_retriever(k: int = 5, meta_filter: dict | None = None):
    vs = load_vectorstore()
    search_kwargs = {"k": k}
    if meta_filter:
        search_kwargs["filter"] = meta_filter  # apply Chroma metadata filter
    return vs.as_retriever(search_kwargs=search_kwargs)
