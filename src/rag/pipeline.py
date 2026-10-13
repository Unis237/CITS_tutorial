"""Compose ingestion, indexing, retrieval, and response generation."""
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from .config import DATA_DIR, CHROMA_DIR
from .embedder import create_embeddings
from .retriever import build_vector_store, create_retriever
from .generator import create_answer_chain, validate_question, add_sources


def build_pipeline(data_dir=DATA_DIR, persist_directory=CHROMA_DIR):
    """Synchronize changed PDFs and return a reusable RAG chain."""
    print("Loading the embedding model...")
    embeddings = create_embeddings()
    vector_store = build_vector_store(embeddings, persist_directory, data_dir)
    retriever = create_retriever(vector_store)
    answer_chain = create_answer_chain()
    return (
        RunnableLambda(validate_question)
        | {
            "passages": retriever,
            "question": RunnablePassthrough(),
        }
        | RunnablePassthrough.assign(answer=answer_chain)
        | RunnableLambda(add_sources)
    )
