"""
EDGAR-Searcher evaluation harness.

Runs the verified golden set (``eval/golden.json``) against the live stack and
reports two independent scores, so a retrieval regression is never mistaken for
a generation regression:

Retrieval quality
    - Recall@1     : is the top-ranked chunk from the expected filing/section?
    - Recall@k     : is any of the top-k chunks from the expected filing/section?
    - Section-precision@k : of the chunks actually returned, what fraction sit in
      one of the expected items? Divided by the number of chunks returned, not
      by k, so a short result list is not penalised as if it were wrong.

Answer accuracy
    - Each question goes through the full RAG path (retrieve -> rerank -> LLM)
      and the generated answer is graded by ``eval/grading.py``: a deterministic,
      LLM-free grader driven by the numbers and phrases recorded in the golden
      set. Reported overall and broken out by coverage dimension.

Filing identity is matched on accession number, falling back to an explicit
fiscal-year field on the chunk metadata. It is never inferred from the filing
date: Apple's FY2024 10-K was FILED on 2024-11-01 for a fiscal year that ended
2024-09-28, and the FY2023 10-K was filed 2023-11-03, so ``filing_date[:4]``
answers a different question than "which fiscal year is this?".

Run with the same env vars the Streamlit app uses (CHROMA_HOST, OLLAMA_HOST).

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --retrieval-only
    python eval/run_eval.py --collection sec_filings_embeddings_v2 --k 5
    python eval/run_eval.py --reranker crossencoder --json eval/results_run1.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grading import GradeResult, grade, parse_numbers  # noqa: E402

# Chunk metadata keys that state a fiscal year directly. `filing_date` is
# deliberately absent: the date a filing was submitted is not its fiscal year.
FISCAL_YEAR_KEYS = (
    "fiscal_year",
    "fiscal_year_end",
    "period_end",
    "period_of_report",
    "report_date",
)

ACCESSION_KEYS = ("accession_number", "accession")

DIMENSION_ORDER = (
    "revenue",
    "mix",
    "segments",
    "margins",
    "opex",
    "profitability",
    "tax",
    "cash_flow",
    "balance_sheet",
    "capital_return",
)


class EvalSetupError(SystemExit):
    """Raised when the stack the eval depends on is not usable."""

    def __init__(self, message: str) -> None:
        super().__init__(f"\nEVAL ABORTED — {message}\n")


def _norm_accession(value: Any) -> str:
    return re.sub(r"[^0-9]", "", str(value or ""))


def _norm_item(value: Any) -> str:
    return str(value or "").strip().upper()


def meta_fiscal_year(meta: Dict[str, Any]) -> Optional[int]:
    """
    Fiscal year of a chunk, from an explicit metadata field only.

    Returns None when the chunk carries no fiscal-year information, rather than
    guessing from the filing date.
    """
    for key in FISCAL_YEAR_KEYS:
        raw = meta.get(key)
        if raw in (None, ""):
            continue
        found = re.search(r"(?:19|20)\d{2}", str(raw))
        if found:
            return int(found.group(0))
    return None


def meta_accession(meta: Dict[str, Any]) -> str:
    for key in ACCESSION_KEYS:
        normalized = _norm_accession(meta.get(key))
        if normalized:
            return normalized
    return ""


def filing_matches(meta: Dict[str, Any], entry: Dict[str, Any]) -> Optional[bool]:
    """
    Whether a chunk comes from the filing the golden entry is about.

    Accession number is exact and preferred. An explicit fiscal-year field is the
    fallback. Returns None when the chunk metadata cannot answer the question —
    those are reported as unresolved rather than silently scored as misses.
    """
    chunk_accession = meta_accession(meta)
    golden_accession = _norm_accession(entry.get("accession"))
    if chunk_accession and golden_accession:
        return chunk_accession == golden_accession

    chunk_year = meta_fiscal_year(meta)
    golden_year = entry.get("fiscal_year")
    if chunk_year is not None and golden_year is not None:
        return chunk_year == int(golden_year)

    return None


def ticker_matches(meta: Dict[str, Any], entry: Dict[str, Any]) -> bool:
    return _norm_item(meta.get("ticker")) == _norm_item(entry.get("ticker"))


def item_matches(meta: Dict[str, Any], entry: Dict[str, Any]) -> bool:
    """True when the chunk's item is one of the entry's acceptable items."""
    expected = [_norm_item(i) for i in entry.get("expected_items") or []]
    return _norm_item(meta.get("item_number")) in expected


def section_matches(meta: Dict[str, Any], entry: Dict[str, Any]) -> bool:
    """Ticker + item only; used for section precision."""
    return ticker_matches(meta, entry) and item_matches(meta, entry)


def full_matches(meta: Dict[str, Any], entry: Dict[str, Any]) -> bool:
    """Ticker + item + confirmed filing identity."""
    if not section_matches(meta, entry):
        return False
    return filing_matches(meta, entry) is True


def score_retrieval(
    metas: Sequence[Dict[str, Any]], entry: Dict[str, Any]
) -> Dict[str, Any]:
    """Retrieval metrics for a single question over the chunks actually returned."""
    returned = len(metas)
    section_hits = sum(1 for m in metas if section_matches(m, entry))
    unresolved = sum(
        1
        for m in metas
        if section_matches(m, entry) and filing_matches(m, entry) is None
    )
    return {
        "returned": returned,
        "hit_at_1": bool(metas) and full_matches(metas[0], entry),
        "hit_at_k": any(full_matches(m, entry) for m in metas),
        # Divide by what actually came back, not by k.
        "section_precision": (section_hits / returned) if returned else 0.0,
        "section_hits": section_hits,
        "unresolved_filing_identity": unresolved,
    }


def expected_summary(entry: Dict[str, Any]) -> str:
    """Short description of what a correct answer must contain."""
    mode = entry.get("grading", "numeric")
    if mode == "categorical":
        return ", ".join(entry.get("expected_tokens") or entry.get("aliases") or [])
    if mode == "keyword":
        parts = list(entry.get("required_keywords") or [])
        if entry.get("answer_numeric") is not None:
            parts.append(f"{entry['answer_numeric']} {entry.get('unit')}")
        return " + ".join(parts)

    summary = f"{entry.get('answer_numeric')} {entry.get('unit')}"
    if entry.get("secondary_numeric") is not None:
        summary += f" + {entry['secondary_numeric']} {entry.get('secondary_unit')}"
    return summary


def actual_summary(answer: str, entry: Dict[str, Any], width: int = 46) -> str:
    """Short description of what the model actually said."""
    if not answer:
        return "(no answer)"
    if entry.get("grading", "numeric") == "numeric":
        numbers = [f"{p.text}" for p in parse_numbers(answer)][:4]
        if numbers:
            return ", ".join(numbers)[:width]
    flat = " ".join(answer.split())
    return flat[:width] + ("…" if len(flat) > width else "")


def load_golden(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load the golden set and validate the invariants the eval relies on."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise EvalSetupError(f"golden set not found: {path}")
    except json.JSONDecodeError as exc:
        raise EvalSetupError(f"golden set is not valid JSON ({path}): {exc}")

    if isinstance(data, list):
        raise EvalSetupError(
            f"{path} uses the old flat-list format. The golden set must be an "
            "object with 'metadata' and 'questions' keys."
        )

    questions = data.get("questions")
    if not questions:
        raise EvalSetupError(f"golden set has no questions: {path}")

    ids = [q.get("id") for q in questions]
    if len(set(ids)) != len(ids):
        raise EvalSetupError(f"golden set has duplicate ids: {path}")
    for question in questions:
        if not question.get("expected_items"):
            raise EvalSetupError(f"{question.get('id')} has no expected_items")

    return data.get("metadata", {}), questions


def build_reranker(name: str):
    """Reranker instance for the retrieval-metric path (None when disabled)."""
    from backend.reranker import CrossEncoderReranker, Reranker

    if name == "off":
        return None
    if name == "bm25":
        return Reranker()
    if name == "crossencoder":
        return CrossEncoderReranker()
    raise EvalSetupError(f"unknown reranker: {name}")


def _ollama_model_names(listing: Any) -> List[str]:
    models = getattr(listing, "models", None)
    if models is None and isinstance(listing, dict):
        models = listing.get("models")
    names = []
    for model in models or []:
        name = getattr(model, "model", None) or getattr(model, "name", None)
        if name is None and isinstance(model, dict):
            name = model.get("model") or model.get("name")
        if name:
            names.append(str(name))
    return names


def preflight(collection_name: str, need_llm: bool, llm_model: str):
    """
    Verify Chroma and Ollama are usable before scoring anything.

    Without this an outage looks exactly like a model that gets every question
    wrong, and the eval quietly reports 0%.
    """
    chroma_host = os.environ.get("CHROMA_HOST", "http://localhost:8000")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    try:
        from backend.embedding_client import EmbeddingClient
    except ImportError as exc:
        raise EvalSetupError(
            f"cannot import the backend ({exc}). "
            "Install dependencies first: pip install -r requirements.txt"
        )

    client = EmbeddingClient()

    try:
        client.chroma_client.heartbeat()
    except Exception as exc:
        raise EvalSetupError(
            f"Chroma is unreachable at CHROMA_HOST={chroma_host} ({exc}). "
            "Start it with: chroma run --path ./chroma_data --port 8000"
        )

    try:
        collection = client.chroma_client.get_collection(collection_name)
        count = collection.count()
    except Exception as exc:
        raise EvalSetupError(
            f"Chroma collection {collection_name!r} is not available ({exc}). "
            "Load filings from the app's home page, or pass --collection."
        )
    if count == 0:
        raise EvalSetupError(
            f"Chroma collection {collection_name!r} is empty — load the filing "
            "the golden set covers before running the eval."
        )

    try:
        client.ollama_client.embed(model=client.model, input="preflight")
    except Exception as exc:
        raise EvalSetupError(
            f"Ollama embedding model {client.model!r} is unavailable at "
            f"OLLAMA_HOST={ollama_host} ({exc}). "
            f"Start Ollama and run: ollama pull {client.model}"
        )

    if need_llm:
        try:
            listing = client.ollama_client.list()
        except Exception as exc:
            raise EvalSetupError(
                f"Ollama is unreachable at OLLAMA_HOST={ollama_host} ({exc}). "
                "Start it with: ollama serve — or pass --retrieval-only."
            )
        names = _ollama_model_names(listing)
        if names and not any(
            n == llm_model or n.startswith(f"{llm_model}:") for n in names
        ):
            raise EvalSetupError(
                f"Ollama does not have the answering model {llm_model!r} "
                f"(available: {', '.join(sorted(names))}). "
                f"Run: ollama pull {llm_model} — or pass --retrieval-only."
            )

    return client, count


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    golden_path = Path(args.golden)
    metadata, questions = load_golden(golden_path)

    client, chunk_count = preflight(
        args.collection, not args.retrieval_only, args.model
    )
    reranker = build_reranker(args.reranker)

    llm = None
    if not args.retrieval_only:
        from backend.llm_client import LLMClient

        llm = LLMClient(model=args.model)

    rows: List[Dict[str, Any]] = []
    generation_errors: List[str] = []

    for entry in questions:
        question = entry["question"]
        fetch_count = args.candidates if reranker is not None else args.k

        try:
            results = client.query(
                query=question,
                collection_name=args.collection,
                n_results=fetch_count,
                include_metadata=True,
            )
        except Exception as exc:
            raise EvalSetupError(
                f"retrieval failed on {entry['id']} ({exc}). "
                "Chroma or the embedding model became unavailable mid-run."
            )

        if reranker is not None and results.get("documents"):
            results = reranker.rerank(question, results, n_final=args.k)

        metas = list(results.get("metadatas", []))[: args.k]
        retrieval = score_retrieval(metas, entry)

        answer: Optional[str] = None
        result: Optional[GradeResult] = None
        error: Optional[str] = None
        if llm is not None:
            try:
                answer = llm.ask(
                    question,
                    use_rag=True,
                    collection_name=args.collection,
                    n_results=args.k,
                    n_candidates=args.candidates,
                    reranker_mode=args.reranker,
                )
                result = grade(answer, entry)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                error = f"{type(exc).__name__}: {exc}"
                generation_errors.append(f"{entry['id']}: {error}")

        rows.append(
            {
                "id": entry["id"],
                "dimension": entry.get("dimension", "uncategorised"),
                "question": question,
                "expected": expected_summary(entry),
                "expected_items": entry.get("expected_items"),
                "fiscal_year": entry.get("fiscal_year"),
                "accession": entry.get("accession"),
                "grading": entry.get("grading", "numeric"),
                "retrieval": retrieval,
                "answer": answer,
                "actual": actual_summary(answer or "", entry),
                "answer_passed": bool(result) if result is not None else None,
                "reasons": result.reasons if result is not None else [],
                "error": error,
            }
        )

    graded = [r for r in rows if r["answer_passed"] is not None]
    if llm is not None and not graded:
        raise EvalSetupError(
            "every question failed to generate an answer — the LLM is not "
            "usable, so no accuracy score is meaningful. First error: "
            + (generation_errors[0] if generation_errors else "unknown")
        )

    return {
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "collection": args.collection,
            "chunks_in_collection": chunk_count,
            "k": args.k,
            "candidates": args.candidates,
            "reranker": args.reranker,
            "model": None if args.retrieval_only else args.model,
            "retrieval_only": args.retrieval_only,
            "golden": str(golden_path),
            "chroma_host": os.environ.get("CHROMA_HOST", "http://localhost:8000"),
            "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        },
        "filing": metadata,
        "retrieval": summarise_retrieval(rows),
        "answers": summarise_answers(rows),
        "generation_errors": generation_errors,
        "questions": rows,
    }


def summarise_retrieval(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    recall_at_1 = sum(1 for r in rows if r["retrieval"]["hit_at_1"])
    recall_at_k = sum(1 for r in rows if r["retrieval"]["hit_at_k"])
    precision = sum(r["retrieval"]["section_precision"] for r in rows)
    empty = sum(1 for r in rows if r["retrieval"]["returned"] == 0)
    unresolved = sum(r["retrieval"]["unresolved_filing_identity"] for r in rows)
    return {
        "questions": total,
        "recall_at_1": recall_at_1,
        "recall_at_1_pct": recall_at_1 / total if total else 0.0,
        "recall_at_k": recall_at_k,
        "recall_at_k_pct": recall_at_k / total if total else 0.0,
        "section_precision_at_k": precision / total if total else 0.0,
        "questions_with_no_results": empty,
        "chunks_with_unresolved_filing_identity": unresolved,
    }


def summarise_answers(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    graded = [r for r in rows if r["answer_passed"] is not None]
    if not graded:
        return {"graded": 0, "correct": 0, "accuracy": None, "by_dimension": {}}

    correct = sum(1 for r in graded if r["answer_passed"])
    by_dimension: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    seen = [r["dimension"] for r in graded]
    ordered = [d for d in DIMENSION_ORDER if d in seen]
    ordered += sorted({d for d in seen if d not in DIMENSION_ORDER})
    for dimension in ordered:
        subset = [r for r in graded if r["dimension"] == dimension]
        hits = sum(1 for r in subset if r["answer_passed"])
        by_dimension[dimension] = {
            "graded": len(subset),
            "correct": hits,
            "accuracy": hits / len(subset) if subset else 0.0,
        }

    return {
        "graded": len(graded),
        "correct": correct,
        "accuracy": correct / len(graded),
        "errors": sum(1 for r in rows if r["error"]),
        "by_dimension": by_dimension,
    }


def report(results: Dict[str, Any]) -> None:
    run = results["run"]
    filing = results.get("filing", {})
    retrieval = results["retrieval"]
    answers = results["answers"]
    rows = results["questions"]

    print()
    print("=" * 100)
    print("EDGAR-Searcher evaluation")
    print("=" * 100)
    print(
        f"  Filing     : {filing.get('company', '?')} {filing.get('form', '')} "
        f"FY{filing.get('fiscal_year', '?')} "
        f"(period end {filing.get('period_end', '?')}, "
        f"filed {filing.get('filed_date', '?')})"
    )
    print(f"  Accession  : {filing.get('accession', '?')}")
    print(
        f"  Collection : {run['collection']} " f"({run['chunks_in_collection']} chunks)"
    )
    print(
        f"  Settings   : k={run['k']} candidates={run['candidates']} "
        f"reranker={run['reranker']} model={run['model'] or '(retrieval-only)'}"
    )

    print()
    print("-- RETRIEVAL " + "-" * 87)
    total = retrieval["questions"]
    width = 22
    recall_k_label = "Recall@{}".format(run["k"])
    precision_label = "Section-precision@{}".format(run["k"])
    print(
        f"  {'Recall@1':<{width}}: {retrieval['recall_at_1']}/{total} = "
        f"{retrieval['recall_at_1_pct']:.0%}"
    )
    print(
        f"  {recall_k_label:<{width}}: {retrieval['recall_at_k']}/{total} = "
        f"{retrieval['recall_at_k_pct']:.0%}"
    )
    print(
        f"  {precision_label:<{width}}: "
        f"{retrieval['section_precision_at_k']:.2f}  "
        "(over chunks actually returned)"
    )
    if retrieval["questions_with_no_results"]:
        print(
            f"  ! {retrieval['questions_with_no_results']} question(s) returned "
            "no chunks at all"
        )
    if retrieval["chunks_with_unresolved_filing_identity"]:
        print(
            f"  ! {retrieval['chunks_with_unresolved_filing_identity']} chunk(s) "
            "carry no accession or fiscal-year metadata, so their filing "
            "identity could not be confirmed and they counted as misses"
        )

    print()
    print("-- ANSWER ACCURACY " + "-" * 81)
    if answers["graded"] == 0:
        print("  not scored (--retrieval-only)")
    else:
        print(
            f"  {'Overall':<22}: {answers['correct']}/{answers['graded']} = "
            f"{answers['accuracy']:.0%}"
        )
        if answers.get("errors"):
            print(f"  ! {answers['errors']} question(s) errored during generation")
        print()
        for dimension, stats in answers["by_dimension"].items():
            bar = "#" * round(stats["accuracy"] * 20)
            print(
                f"    {dimension:<16} {stats['correct']}/{stats['graded']} "
                f"{stats['accuracy']:>4.0%}  {bar}"
            )

    print()
    print("-- PER QUESTION " + "-" * 84)
    header = (
        f"  {'ID':<5} {'DIMENSION':<15} {'R@1':<4} {'R@k':<4} {'SP@k':<6} "
        f"{'EXPECTED':<30} {'ACTUAL':<28} RESULT"
    )
    print(header)
    print("  " + "-" * 96)
    for row in rows:
        ret = row["retrieval"]
        if row["answer_passed"] is None:
            verdict = "n/a" if not row["error"] else "ERROR"
        else:
            verdict = "PASS" if row["answer_passed"] else "FAIL"
        print(
            f"  {row['id']:<5} {row['dimension']:<15} "
            f"{'Y' if ret['hit_at_1'] else '.':<4} "
            f"{'Y' if ret['hit_at_k'] else '.':<4} "
            f"{ret['section_precision']:<6.2f} "
            f"{row['expected'][:30]:<30} {row['actual'][:28]:<28} {verdict}"
        )

    failures = [r for r in rows if r["answer_passed"] is False or r["error"]]
    if failures:
        print()
        print("-- FAILURE DETAIL " + "-" * 82)
        for row in failures:
            print(f"  {row['id']}  {row['question']}")
            print(f"      expected : {row['expected']}")
            if row["error"]:
                print(f"      error    : {row['error']}")
            else:
                answer = " ".join((row["answer"] or "").split())
                print(f"      answer   : {answer[:300]}")
                for reason in row["reasons"]:
                    print(f"      why fail : {reason}")
            print()
    print()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval quality and end-to-end answer accuracy against "
            "the verified golden set."
        )
    )
    parser.add_argument(
        "--golden",
        default=str(Path(__file__).parent / "golden.json"),
        help="path to the golden set (default: eval/golden.json)",
    )
    parser.add_argument(
        "--collection",
        default="sec_filings_embeddings_v2",
        help="Chroma collection to query",
    )
    parser.add_argument("--k", type=int, default=5, help="top-k cut after rerank")
    parser.add_argument(
        "--candidates", type=int, default=10, help="initial fetch before rerank"
    )
    parser.add_argument(
        "--reranker", default="off", choices=("off", "bm25", "crossencoder")
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "llama2"),
        help="Ollama model used to generate answers (env: OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="skip answer generation and grading; no LLM required",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="write machine-readable results to this path for run tracking",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.k < 1:
        raise EvalSetupError("--k must be at least 1")
    if args.candidates < args.k:
        raise EvalSetupError("--candidates must be >= --k")

    results = evaluate(args)
    report(results)

    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, default=str))
        print(f"  wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
