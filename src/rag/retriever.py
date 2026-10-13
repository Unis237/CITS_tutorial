"""Persist document embeddings and synchronize only changed PDFs."""
import hashlib
import json
from pathlib import Path

from filelock import FileLock
from langchain_chroma import Chroma

from .chunker import chunk_documents
from .config import (
    CHROMA_DIR, COLLECTION_NAME, RETRIEVAL_K, DATA_DIR,
    EMBEDDING_MODEL, EMBEDDING_REVISION, CHUNK_SIZE, CHUNK_OVERLAP,
)
from .loader import load_documents


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_vector_store(embeddings, persist_directory=CHROMA_DIR, data_dir=DATA_DIR):
    """Check PDFs at startup; retain unchanged vectors across process restarts."""
    directory = Path(persist_directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(COLLECTION_NAME.encode()).hexdigest()[:16]
    manifest_path = directory / f"index-{key}.json"
    # Prevent two script startups from synchronizing this collection together.
    with FileLock(str(manifest_path) + ".lock"):
        store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(directory),
        )
        settings = {
            "version": 1,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "normalize_embeddings": True,
            "document_prompt": "",
        }
        try:
            previous = json.loads(manifest_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            previous = {}
        if not isinstance(previous, dict):
            previous = {}
        rebuild = previous.get("settings") != settings
        old_files = previous.get("files", {}) if not rebuild else {}
        current_ids = set(store.get(include=[])["ids"])
        files = {}
        pending = []
        for path in sorted(Path(data_dir).glob("*.pdf")):
            digest = file_hash(path)
            old = old_files.get(path.name, {})
            if old.get("hash") == digest and old.get("ids") and set(old["ids"]) <= current_ids:
                files[path.name] = old
                continue
            chunks = chunk_documents(load_documents(pdf_files=[path]))
            if file_hash(path) != digest:
                raise RuntimeError(f"{path.name} changed during indexing; retry when editing is finished.")
            prefix = hashlib.sha256((path.name + digest).encode()).hexdigest()
            ids = [f"{prefix}:{i}" for i in range(len(chunks))]
            files[path.name] = {"hash": digest, "ids": ids}
            pending.append((chunks, ids))

        # Legacy indexes need one initial rebuild. Reset also permits an
        # embedding model with a different vector dimension.
        if rebuild:
            store.reset_collection()
        for chunks, ids in pending:
            # Stable IDs avoid duplicates when retrying an interrupted update.
            for start in range(0, len(ids), 100):
                store.add_documents(chunks[start:start + 100], ids=ids[start:start + 100])
        wanted_ids = {identifier for entry in files.values() for identifier in entry["ids"]}
        stale_ids = sorted(set(store.get(include=[])["ids"]) - wanted_ids)
        for start in range(0, len(stale_ids), 100):
            store.delete(ids=stale_ids[start:start + 100])
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"settings": settings, "files": files}, indent=2))
        temporary.replace(manifest_path)
        if not files:
            raise FileNotFoundError(f"Add PDFs to {data_dir}.")
        print(f"Index ready: {len(pending)} new or changed PDFs; {len(files) - len(pending)} reused.")
        return store


def create_retriever(vector_store):
    return vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
