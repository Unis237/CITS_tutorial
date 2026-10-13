"""Download the pinned embedding model and pull the Ollama chat model."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag.config import EMBEDDING_MODEL, EMBEDDING_REVISION, OLLAMA_MODEL, OLLAMA_URL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    ollama = shutil.which("ollama")
    if ollama is None:
        parser.exit(1, "Install Ollama from https://ollama.com/download and rerun this script.\n")

    from huggingface_hub import snapshot_download

    print(f"Downloading {EMBEDDING_MODEL}...", flush=True)
    snapshot_download(repo_id=EMBEDDING_MODEL, revision=EMBEDDING_REVISION)
    print(f"Pulling {OLLAMA_MODEL}...", flush=True)
    try:
        subprocess.run(
            [ollama, "pull", OLLAMA_MODEL],
            check=True,
            env={**os.environ, "OLLAMA_HOST": OLLAMA_URL},
        )
    except subprocess.CalledProcessError:
        parser.exit(1, "Ollama pull failed. Check that Ollama is running (ollama serve) and retry.\n")
    print("Models are ready. Run scripts/ask.py with a question.")


if __name__ == "__main__":
    main()
