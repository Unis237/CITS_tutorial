"""Load the pinned embedding model from the local cache on the CPU."""
from langchain_huggingface import HuggingFaceEmbeddings
from .config import EMBEDDING_MODEL, EMBEDDING_REVISION


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "revision": EMBEDDING_REVISION,
            "device": "cpu",
            "local_files_only": True,
        },
        encode_kwargs={"normalize_embeddings": True, "batch_size": 2, "prompt": ""},
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt": "Instruct: Find document passages that answer the question.\nQuery: ",
        },
    )
