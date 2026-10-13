"""Ask a question about the PDFs in data/."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to answer (enclose it in quotes).")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("Question must not be empty.")

    from rag.pipeline import build_pipeline

    rag_chain = build_pipeline()
    print(rag_chain.invoke(args.question))


if __name__ == "__main__":
    main()
