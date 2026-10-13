"""Split pages into overlapping passages while retaining source metadata."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError("The PDFs contain no readable text. Scanned PDFs may need OCR first.")
    
    return chunks
