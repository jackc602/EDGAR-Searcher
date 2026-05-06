"""
Lightweight retrieval-quality evaluator.

Reads eval/golden.json (list of {question, ticker, expected_item, expected_year}),
runs each question through EmbeddingClient.query against the Chroma collection
the app populates, and reports:

- Recall@1 / Recall@5 (item-level): does any of the top-k retrieved chunks have
  matching (ticker, item_number, year)?
- Section-precision@5: of the top-5 retrieved, what fraction match the expected
  item for that question?

Run with the same env vars the Streamlit app uses (CHROMA_HOST, OLLAMA_HOST).
The golden set is small by design — populate it with questions whose answers
you know are present in your loaded filings.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --collection sec_filings_embeddings_v2 --k 5
    python eval/run_eval.py --reranker crossencoder
"""
import argparse
import json
import sys
from pathlib import Path

# Make `backend` importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.embedding_client import EmbeddingClient  # noqa: E402
from backend.reranker import CrossEncoderReranker, Reranker  # noqa: E402


def _matches(meta: dict, ticker: str, item: str, year: str) -> bool:
    if (meta.get("ticker") or "").upper() != ticker.upper():
        return False
    if (meta.get("item_number") or "").upper() != item.upper():
        return False
    return (meta.get("filing_date") or "")[:4] == year


def _matches_item_only(meta: dict, ticker: str, item: str) -> bool:
    if (meta.get("ticker") or "").upper() != ticker.upper():
        return False
    return (meta.get("item_number") or "").upper() == item.upper()


def _build_reranker(name: str):
    if name == "off":
        return None
    if name == "bm25":
        return Reranker()
    if name == "crossencoder":
        return CrossEncoderReranker()
    raise SystemExit(f"unknown reranker: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(Path(__file__).parent / "golden.json"))
    parser.add_argument("--collection", default="sec_filings_embeddings_v2")
    parser.add_argument("--k", type=int, default=5, help="top-k cut after rerank")
    parser.add_argument("--candidates", type=int, default=10, help="initial fetch")
    parser.add_argument(
        "--reranker", default="off", choices=("off", "bm25", "crossencoder")
    )
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text())
    if not golden:
        raise SystemExit("golden set is empty — populate eval/golden.json")

    client = EmbeddingClient()
    reranker = _build_reranker(args.reranker)

    recall_at_1 = 0
    recall_at_k = 0
    section_precision_sum = 0.0

    rows = []
    for entry in golden:
        question = entry["question"]
        ticker = entry["ticker"]
        item = entry["expected_item"]
        year = entry["expected_year"]

        fetch_count = args.candidates if reranker is not None else args.k
        results = client.query(
            query=question,
            collection_name=args.collection,
            n_results=fetch_count,
            include_metadata=True,
        )
        if reranker is not None and results.get("documents"):
            results = reranker.rerank(question, results, n_final=args.k)

        metas = results.get("metadatas", [])[: args.k]

        hit_at_1 = bool(metas) and _matches(metas[0], ticker, item, year)
        hit_at_k = any(_matches(m, ticker, item, year) for m in metas)
        section_hits = sum(1 for m in metas if _matches_item_only(m, ticker, item))
        section_precision = section_hits / args.k if args.k else 0.0

        recall_at_1 += int(hit_at_1)
        recall_at_k += int(hit_at_k)
        section_precision_sum += section_precision

        rows.append({
            "question": question,
            "expected": f"{ticker} Item {item} ({year})",
            "hit@1": hit_at_1,
            "hit@k": hit_at_k,
            "section_precision@k": round(section_precision, 2),
        })

    n = len(golden)
    print(f"\nGolden set: {n} questions, k={args.k}, reranker={args.reranker}")
    print(f"  Recall@1            : {recall_at_1}/{n} = {recall_at_1 / n:.0%}")
    print(f"  Recall@{args.k:<2}           : {recall_at_k}/{n} = {recall_at_k / n:.0%}")
    print(f"  Section-precision@{args.k}: {section_precision_sum / n:.2f}")
    print()
    for row in rows:
        flag = "✓" if row["hit@k"] else "✗"
        print(f"  {flag} {row['expected']:<30}  prec@k={row['section_precision@k']}  {row['question'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
