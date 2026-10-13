"""Load PDF pages and retain filenames and one-based viewer page numbers."""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from .config import DATA_DIR


def load_documents(data_dir=DATA_DIR, pdf_files=None):
    pdf_files = sorted(Path(data_dir).glob("*.pdf")) if pdf_files is None else list(pdf_files)
    if not pdf_files:
        raise FileNotFoundError(f"Add PDFs to {data_dir}.")
    
    pages = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file), mode="page")
        for page in loader.load():
            page.metadata = {
                "source": pdf_file.name,
                "page": page.metadata["page"] + 1,
            }
            pages.append(page)
    
    print(f"Loaded {len(pages)} pages from {len(pdf_files)} PDFs.")
    return pages
