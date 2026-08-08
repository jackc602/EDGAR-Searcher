"""
Unit tests for the deterministic eval grader.

These run with no Chroma, no Ollama and no network — they only import
``eval/grading.py`` and read ``eval/golden.json`` from disk.
"""

import pytest

from grading import (
    grade,
    grade_categorical,
    grade_keyword,
    grade_numeric,
    normalize_value,
    parse_number,
    parse_numbers,
)

# --------------------------------------------------------------------------
# parse_number
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("391,035", [391035.0]),
        ("1,234,567", [1234567.0]),
        ("$391,035 million", [391035000000.0]),
        ("$6.08", [6.08]),
        ("46.2%", [46.2]),
        ("46.2 percent", [46.2]),
        ("$391.0 billion", [391000000000.0]),
        ("391 billion", [391000000000.0]),
        ("$95.0B", [95000000000.0]),
        ("$10.2 billion", [10200000000.0]),
        ("57,467M", [57467000000.0]),
        ("499 million shares", [499000000.0]),
        ("(8)%", [-8.0]),
        ("(6)%", [-6.0]),
        ("(19,154)", [-19154.0]),
        ("-3.4%", [-3.4]),
        ("no numbers here", []),
        ("", []),
    ],
)
def test_parse_number_forms(text, expected):
    assert parse_number(text) == expected


def test_parse_number_finds_every_number_in_a_sentence():
    text = "Net sales rose 2% from $383,285 million to $391,035 million."
    assert parse_number(text) == [2.0, 383285000000.0, 391035000000.0]


def test_surrounding_parenthesis_does_not_flip_the_sign():
    """'(approximately $391.0 billion)' is positive; only '(8)' is negative."""
    assert parse_number("(approximately $391.0 billion)") == [391000000000.0]
    assert parse_number("Diluted EPS was $6.08 (basic EPS $6.11)") == [6.08, 6.11]


def test_scale_word_is_not_matched_inside_a_word():
    """The 'b' in 'basic' must not be read as a billions suffix."""
    assert parse_number("6.08 basic") == [6.08]
    assert parse_number("29 million") == [29000000.0]


def test_parsed_number_records_explicit_scale_and_precision():
    bare, scaled = parse_numbers("391,035 and 391.0 billion")
    assert bare.explicit_scale is False
    assert bare.rounding_tolerance == 0.0
    assert scaled.explicit_scale is True
    # "391.0 billion" is only resolvable to half of its last digit.
    assert scaled.rounding_tolerance == pytest.approx(0.05e9)


# --------------------------------------------------------------------------
# unit normalisation
# --------------------------------------------------------------------------


def test_normalize_value_puts_units_on_a_common_base():
    assert normalize_value(391035, "usd_millions") == normalize_value(
        391.035, "usd_billions"
    )
    assert normalize_value(499, "shares_millions") == 499e6
    assert normalize_value(46.2, "percent") == 46.2


def test_normalize_value_rejects_an_unknown_unit():
    with pytest.raises(ValueError):
        normalize_value(1, "furlongs")


def test_billions_answer_matches_a_millions_golden_value(entries):
    """391.0 billion == 391,035 million, within the precision of '391.0'."""
    q01 = entries["q01"]
    assert grade_numeric("Total net sales were $391.0 billion.", q01)
    assert grade_numeric("Total net sales were 391,035 million.", q01)
    assert grade_numeric("Total net sales were $391,035 million.", q01)
    assert grade_numeric("Total net sales were 391 billion.", q01)


def test_millions_answer_matches_a_billions_golden_value(entries):
    """Q20 is stored in billions; the cash-flow statement quotes $94,949M."""
    q20 = entries["q20"]
    assert grade_numeric("Apple repurchased 499 million shares for $95.0 billion.", q20)
    assert grade_numeric("Apple repurchased $94,949 million (499 million shares).", q20)


def test_a_wrong_magnitude_still_fails(entries):
    q01 = entries["q01"]
    assert not grade_numeric("Total net sales were $391.0 million.", q01)
    assert not grade_numeric("Total net sales were $383,285 million.", q01)


# --------------------------------------------------------------------------
# tolerance boundaries
# --------------------------------------------------------------------------


def test_tolerance_boundary_just_inside_and_just_outside(entries):
    """Q09 is 46.2% with tolerance 0.05, so 46.25 is in and 46.26 is out."""
    q09 = entries["q09"]
    assert grade_numeric("Gross margin was 46.25%.", q09)
    assert grade_numeric("Gross margin was 46.15%.", q09)
    assert not grade_numeric("Gross margin was 46.26%.", q09)
    assert not grade_numeric("Gross margin was 46.14%.", q09)


def test_millions_tolerance_boundary(entries):
    """Q01 is 391,035 with tolerance 1 (million)."""
    q01 = entries["q01"]
    assert grade_numeric("Total net sales were 391,036 million.", q01)
    assert not grade_numeric("Total net sales were 391,037 million.", q01)


def test_eps_tolerance_rejects_basic_eps(entries):
    """Q15 wants diluted EPS 6.08; 6.11 is basic EPS and must not pass."""
    q15 = entries["q15"]
    assert grade_numeric("Diluted EPS was $6.08.", q15)
    assert not grade_numeric("EPS was $6.11.", q15)
    assert not grade_numeric("EPS was $6.13.", q15)


def test_secondary_numeric_must_also_match(entries):
    """Q03 needs both $96,169M and the 13% change."""
    q03 = entries["q03"]
    assert grade_numeric("Services were $96,169 million, up 13%.", q03)
    assert not grade_numeric("Services were $96,169 million.", q03)
    assert not grade_numeric("Services were $96,169 million, up 9%.", q03)


def test_percent_is_compared_on_magnitude(entries):
    """Q06's secondary is -8; a natural answer writes 'down 8%'."""
    q06 = entries["q06"]
    assert grade_numeric("Greater China was $66,952 million, down 8%.", q06)
    assert grade_numeric("Greater China was $66,952 million, (8)%.", q06)
    assert not grade_numeric("Greater China was $66,952 million, down 5%.", q06)


def test_grade_numeric_rejects_an_empty_answer(entries):
    assert not grade_numeric("", entries["q01"])
    assert not grade_numeric("   ", entries["q01"])


# --------------------------------------------------------------------------
# grading modes
# --------------------------------------------------------------------------


def test_numeric_mode(entries):
    q13 = entries["q13"]
    assert grade("Operating income was $123,216 million.", q13).passed
    assert not grade("Operating income was $114,301 million.", q13).passed


def test_categorical_mode_requires_the_expected_token(entries):
    q05 = entries["q05"]
    assert grade_categorical("Services grew fastest, at 13%.", q05)
    assert not grade_categorical("The Mac category grew fastest.", q05)


def test_keyword_mode_requires_every_keyword(entries):
    q17 = entries["q17"]
    assert grade_keyword("Because of the State Aid Decision charge.", q17)
    assert not grade_keyword("Because of a one-time charge in Ireland.", q17)


def test_keyword_mode_also_enforces_the_number(entries):
    """Q17 sets require_numeric, so 'State Aid' alone is not enough."""
    q17 = entries["q17"]
    assert grade("A one-time State Aid charge of $10.2 billion.", q17).passed
    assert grade("The State Aid Decision added $10,246 million.", q17).passed
    assert not grade("A one-time State Aid charge.", q17).passed
    assert not grade("A one-time State Aid charge of $5.0 billion.", q17).passed


def test_unknown_grading_mode_is_rejected():
    with pytest.raises(ValueError):
        grade("anything", {"grading": "vibes"})


def test_empty_answer_fails_every_mode(entries):
    for qid in ("q01", "q05", "q17"):
        result = grade("", entries[qid])
        assert not result.passed
        assert result.reasons


# --------------------------------------------------------------------------
# Q7 — naming the wrong segment must fail
# --------------------------------------------------------------------------


def test_q07_accepts_greater_china(entries):
    q07 = entries["q07"]
    assert grade("Greater China's net sales declined 8% in fiscal 2024.", q07).passed
    assert grade(
        "Greater China was the only segment whose net sales declined, while "
        "Americas, Europe, Japan and Rest of Asia Pacific all grew.",
        q07,
    ).passed


@pytest.mark.parametrize(
    "answer",
    [
        "Europe saw net sales decline in fiscal 2024.",
        "Americas net sales declined during 2024.",
        "Japan's net sales decreased year over year.",
        "Rest of Asia Pacific net sales fell in fiscal 2024.",
        "Europe saw net sales decline in fiscal 2024, unlike Greater China.",
    ],
)
def test_q07_fails_when_another_segment_is_named_as_the_decliner(entries, answer):
    assert not grade(answer, entries["q07"]).passed


def test_q07_fails_when_no_segment_is_named(entries):
    assert not grade("One segment declined during fiscal 2024.", entries["q07"]).passed


# --------------------------------------------------------------------------
# direction checks — a right number with a wrong direction scores zero
# --------------------------------------------------------------------------


def test_q04_iphone_flat_direction(entries):
    """Q04: 201,183 is right, but 'grew' is wrong — the filing says flat."""
    q04 = entries["q04"]
    right = "iPhone net sales were $201,183 million, essentially flat versus 2023."
    assert grade(right, q04).passed

    for wrong in (
        "iPhone net sales grew to $201,183 million in fiscal 2024.",
        "iPhone net sales increased to $201,183 million.",
        "iPhone revenue rose to $201,183 million.",
        "There was growth in iPhone, to $201,183 million.",
    ):
        result = grade(wrong, q04)
        assert not result.passed, wrong
        assert grade_numeric(wrong, q04), "the number itself is right"


def test_q04_requires_the_flat_claim_to_be_stated(entries):
    """Q04 asks 'did they grow?', so silence on direction is not an answer."""
    assert not grade("iPhone net sales were $201,183 million.", entries["q04"]).passed


def test_q14_net_income_down_direction(entries):
    """Q14: net income FELL to 93,736 — claiming an increase scores zero."""
    q14 = entries["q14"]
    right = "Net income was $93,736 million, down 3.4% from $96,995 million."
    assert grade(right, q14).passed

    for wrong in (
        "Net income increased 3.4% to $93,736 million.",
        "Net income grew to $93,736 million, up 3.4%.",
        "Record net income of $93,736 million, 3.4% higher.",
        "Net income rose to $93,736 million, a 3.4% improvement.",
    ):
        result = grade(wrong, q14)
        assert not result.passed, wrong


def test_q14_requires_the_comparison_direction(entries):
    assert not grade("Net income was $93,736 million.", entries["q14"]).passed


def test_q19_equity_declined_direction(entries):
    """Q19: both numbers right, but equity did not increase — 62,146 -> 56,950."""
    q19 = entries["q19"]
    right = (
        "Total assets were $364,980 million and total shareholders' equity was "
        "$56,950 million, down from $62,146 million."
    )
    assert grade(right, q19).passed

    for wrong in (
        "Total assets were $364,980 million and equity increased to $56,950 million.",
        "Total assets $364,980 million; shareholders' equity grew to $56,950 million.",
        "Assets of $364,980 million, with higher shareholders' equity "
        "of $56,950 million.",
    ):
        result = grade(wrong, q19)
        assert not result.passed, wrong
        assert grade_numeric(wrong, q19), "the numbers themselves are right"


def test_q19_does_not_demand_a_direction_word(entries):
    """Q19 asks only for two balances, so a bare correct answer passes."""
    assert grade(
        "Total assets were $364,980 million and shareholders' equity was "
        "$56,950 million.",
        entries["q19"],
    ).passed


def test_direction_failure_is_explained(entries):
    result = grade("Net income increased to $93,736 million.", entries["q14"])
    assert any("direction" in reason for reason in result.reasons)


# --------------------------------------------------------------------------
# golden-set integrity
# --------------------------------------------------------------------------


def test_golden_set_has_twenty_unique_questions(golden):
    questions = golden["questions"]
    assert len(questions) == 20
    assert len({q["id"] for q in questions}) == 20
    assert [q["id"] for q in questions] == [f"q{i:02d}" for i in range(1, 21)]


def test_golden_metadata_describes_the_filing(golden):
    metadata = golden["metadata"]
    assert metadata["accession"] == "0000320193-24-000123"
    assert metadata["ticker"] == "AAPL"
    assert metadata["cik"] == "0000320193"
    assert metadata["form"] == "10-K"
    assert metadata["fiscal_year"] == 2024
    assert metadata["period_end"] == "2024-09-28"
    assert metadata["filed_date"] == "2024-11-01"
    # The fiscal year and the filing date are recorded as separate facts.
    assert metadata["filed_date"] != metadata["period_end"]


def test_every_entry_carries_the_required_fields(golden):
    for entry in golden["questions"]:
        assert entry["ticker"] == "AAPL"
        assert entry["cik"] == "0000320193"
        assert entry["accession"] == "0000320193-24-000123"
        assert entry["fiscal_year"] == 2024
        assert entry["period_end"] == "2024-09-28"
        assert isinstance(entry["expected_items"], list) and entry["expected_items"]
        assert entry["grading"] in ("numeric", "categorical", "keyword")
        assert entry["answer_text"]
        assert entry["source"]
        assert entry["aliases"]


def test_every_verified_answer_text_grades_as_correct(golden):
    """The ledger's own wording must pass its own grader."""
    for entry in golden["questions"]:
        result = grade(entry["answer_text"], entry)
        assert result.passed, f"{entry['id']}: {result.reasons}"


def test_coverage_dimensions_match_the_spec(golden):
    by_dimension = {}
    for entry in golden["questions"]:
        by_dimension.setdefault(entry["dimension"], []).append(entry["id"])
    assert by_dimension == {
        "revenue": ["q01", "q02"],
        "mix": ["q03", "q04", "q05"],
        "segments": ["q06", "q07"],
        "margins": ["q08", "q09", "q10"],
        "opex": ["q11", "q12"],
        "profitability": ["q13", "q14", "q15"],
        "tax": ["q16", "q17"],
        "cash_flow": ["q18"],
        "balance_sheet": ["q19"],
        "capital_return": ["q20"],
    }
