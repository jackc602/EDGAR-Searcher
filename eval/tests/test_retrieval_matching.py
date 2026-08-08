"""
Unit tests for the retrieval matcher in ``eval/run_eval.py``.

These cover the two bugs the rewrite fixes — fiscal year conflated with filing
date, and section precision divided by k instead of by the number of chunks
returned — plus the move from a single expected item to a list.

``run_eval`` imports its Chroma/Ollama dependencies lazily inside functions, so
importing the module here needs no backend, no services and no network.
"""

import pytest

from run_eval import (
    filing_matches,
    full_matches,
    item_matches,
    meta_fiscal_year,
    score_retrieval,
    section_matches,
    summarise_retrieval,
)

ENTRY = {
    "id": "q01",
    "ticker": "AAPL",
    "accession": "0000320193-24-000123",
    "fiscal_year": 2024,
    "period_end": "2024-09-28",
    "expected_items": ["7", "8"],
}


def chunk(**overrides):
    """A chunk metadata dict as the app's DocumentChunk writes it."""
    meta = {
        "ticker": "AAPL",
        "item_number": "7",
        "filing_type": "10-K",
        "filing_date": "2024-11-01",
        "accession_number": "0000320193-24-000123",
        "cik": "0000320193",
    }
    meta.update(overrides)
    return meta


# --------------------------------------------------------------------------
# bug 1 — fiscal year is not the filing-date year
# --------------------------------------------------------------------------


def test_fiscal_year_is_never_read_from_the_filing_date():
    """A chunk with only a filing_date has no resolvable fiscal year."""
    meta = chunk(accession_number="")
    assert meta_fiscal_year(meta) is None


def test_fiscal_year_is_read_from_explicit_fields():
    assert meta_fiscal_year({"fiscal_year": 2024}) == 2024
    assert meta_fiscal_year({"fiscal_year": "FY2024"}) == 2024
    assert meta_fiscal_year({"period_end": "2024-09-28"}) == 2024
    assert meta_fiscal_year({"period_of_report": "2024-09-28"}) == 2024


def test_accession_is_the_primary_filing_identity():
    assert filing_matches(chunk(), ENTRY) is True
    assert (
        filing_matches(chunk(accession_number="0000320193-23-000106"), ENTRY) is False
    )


def test_accession_comparison_ignores_formatting():
    assert filing_matches(chunk(accession_number="000032019324000123"), ENTRY) is True


def test_fiscal_year_is_the_fallback_when_there_is_no_accession():
    meta = chunk(accession_number="", period_end="2024-09-28")
    assert filing_matches(meta, ENTRY) is True
    meta = chunk(accession_number="", period_end="2023-09-30")
    assert filing_matches(meta, ENTRY) is False


def test_unresolvable_filing_identity_is_reported_not_guessed():
    """
    The FY2023 10-K was filed 2023-11-03 and the FY2024 10-K on 2024-11-01, so
    the filing date can look like a fiscal year while meaning something else.
    The matcher must say 'unknown', not guess.
    """
    meta = chunk(accession_number="", filing_date="2024-11-01")
    assert filing_matches(meta, ENTRY) is None
    assert full_matches(meta, ENTRY) is False

    scored = score_retrieval([meta], ENTRY)
    assert scored["unresolved_filing_identity"] == 1
    assert scored["hit_at_1"] is False


def test_a_fiscal_2023_chunk_filed_in_2024_does_not_match_fiscal_2024():
    """The trap the old filing_date[:4] matcher fell into, in reverse."""
    meta = chunk(accession_number="", period_end="2023-09-30", filing_date="2024-01-05")
    assert filing_matches(meta, ENTRY) is False
    assert full_matches(meta, ENTRY) is False


# --------------------------------------------------------------------------
# expected_items is a list
# --------------------------------------------------------------------------


@pytest.mark.parametrize("item, expected", [("7", True), ("8", True), ("1A", False)])
def test_any_expected_item_counts_as_a_section_match(item, expected):
    assert item_matches(chunk(item_number=item), ENTRY) is expected


def test_item_matching_is_case_insensitive():
    entry = dict(ENTRY, expected_items=["1a"])
    assert item_matches(chunk(item_number="1A"), entry) is True


def test_a_different_ticker_never_matches():
    assert section_matches(chunk(ticker="MSFT"), ENTRY) is False
    assert full_matches(chunk(ticker="MSFT"), ENTRY) is False


# --------------------------------------------------------------------------
# bug 2 — section precision divides by results returned, not by k
# --------------------------------------------------------------------------


def test_section_precision_divides_by_results_returned():
    """Two chunks back, both in an expected item, is precision 1.0 — not 0.4."""
    metas = [chunk(item_number="7"), chunk(item_number="8")]
    assert score_retrieval(metas, ENTRY)["section_precision"] == pytest.approx(1.0)


def test_section_precision_with_a_partial_hit():
    metas = [chunk(item_number="7"), chunk(item_number="1A")]
    assert score_retrieval(metas, ENTRY)["section_precision"] == pytest.approx(0.5)


def test_section_precision_of_an_empty_result_is_zero_not_an_error():
    scored = score_retrieval([], ENTRY)
    assert scored["section_precision"] == 0.0
    assert scored["returned"] == 0
    assert scored["hit_at_1"] is False
    assert scored["hit_at_k"] is False


def test_recall_at_1_needs_the_top_chunk_to_match():
    metas = [chunk(item_number="1A"), chunk(item_number="7")]
    scored = score_retrieval(metas, ENTRY)
    assert scored["hit_at_1"] is False
    assert scored["hit_at_k"] is True


def test_summarise_retrieval_aggregates_across_questions():
    rows = [
        {"retrieval": score_retrieval([chunk()], ENTRY)},
        {"retrieval": score_retrieval([chunk(item_number="1A")], ENTRY)},
    ]
    summary = summarise_retrieval(rows)
    assert summary["questions"] == 2
    assert summary["recall_at_1"] == 1
    assert summary["recall_at_k"] == 1
    assert summary["section_precision_at_k"] == pytest.approx(0.5)
    assert summary["questions_with_no_results"] == 0
