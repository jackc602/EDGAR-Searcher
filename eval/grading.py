"""
Deterministic, LLM-free grading for the EDGAR-Searcher eval golden set.

Everything in this module is a pure function over ``(model_answer, golden_entry)``.
There is no network access, no Chroma, no Ollama and no import from ``backend``,
so the graders can be unit-tested standalone with ``python -m pytest eval/tests``.

Grading modes (the ``grading`` field of a golden entry)
------------------------------------------------------
``numeric``
    The answer must contain a number matching ``answer_numeric`` within
    ``tolerance`` (and ``secondary_numeric`` within ``secondary_tolerance``
    when the entry declares one).
``categorical``
    The answer must name the expected entity and must not name a competing
    entity as the subject of the claim.
``keyword``
    The answer must contain every phrase in ``required_keywords``.

Every mode additionally runs the *direction* check, which is driven entirely by
two optional golden-entry fields:

``forbidden_phrases``
    If any of these appear in the answer the entry scores zero, even when the
    number is right. This is what makes "net income increased to $93,736
    million" fail Q14.
``required_direction``
    ``{"label": ..., "any_of": [...]}`` — at least one of the listed phrases
    must appear. Only set this on questions that actually ask for a direction.

A ``require_numeric`` flag lets a non-numeric mode also demand the number (Q17
must mention *both* "State Aid" and the $10.2 billion charge).

Unit handling
-------------
Units are normalised to a common base before comparison, so an answer written
in billions is comparable with a golden value stored in millions. A number the
model wrote *with* an explicit scale word is a rounded restatement, so it also
carries the tolerance implied by its own precision: "$391.0 billion" is
resolvable only to +/- 0.05 billion, which is why it matches 391,035 million.
Bare numbers carry no such slack.

Percentages are compared on magnitude. Natural answers write "down 8%", not
"-8%", so the sign of a percentage is graded by the direction machinery above
rather than by the arithmetic comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "ParsedNumber",
    "GradeResult",
    "parse_number",
    "parse_numbers",
    "normalize_text",
    "normalize_value",
    "grade_numeric",
    "grade_categorical",
    "grade_keyword",
    "check_direction",
    "grade",
]

# Multiplier applied by an explicit scale word written in the answer.
SCALE_WORDS: Dict[str, float] = {
    "trillion": 1e12,
    "trillions": 1e12,
    "billion": 1e9,
    "billions": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "million": 1e6,
    "millions": 1e6,
    "mm": 1e6,
    "m": 1e6,
    "thousand": 1e3,
    "thousands": 1e3,
    "k": 1e3,
}

# Multiplier that converts a golden value expressed in `unit` to the base unit
# of its dimension (dollars, shares, percentage points).
UNIT_SCALE: Dict[str, float] = {
    "usd_millions": 1e6,
    "usd_billions": 1e9,
    "shares_millions": 1e6,
    "usd_per_share": 1.0,
    "percent": 1.0,
    "none": 1.0,
}

# Units where a bare number is ambiguous: filings quote dollars in millions in
# the statements and in billions in the narrative, so both are plausible.
MONEY_UNITS = frozenset({"usd_millions", "usd_billions"})

# Units compared on magnitude only; sign is handled by the direction check.
MAGNITUDE_UNITS = frozenset({"percent"})

_NUMBER_RE = re.compile(
    r"(?P<open>\(\s*)?"
    r"(?P<currency>[$€£]\s*)?"
    r"(?P<sign>[-−–])?\s*"
    r"(?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<scale>trillions?|billions?|millions?|thousands?|bn|mm|[bmk])?\b"
    r"\s*(?P<close>\))?"
    r"\s*(?P<percent>%|percent(?:age)?(?:\s+points?)?)?",
    re.IGNORECASE,
)

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})

# Absolute float-noise allowance, scaled to the magnitude being compared, so a
# tolerance boundary test lands on the inclusive side.
_FLOAT_SLACK_REL = 1e-12


@dataclass(frozen=True)
class ParsedNumber:
    """One numeric literal lifted out of a model answer."""

    value: float
    """The literal itself, sign applied, with no scale word factored in."""

    scale: float = 1.0
    """Multiplier from an explicit scale word ('billion' -> 1e9)."""

    explicit_scale: bool = False
    """True when the answer actually wrote a scale word next to the number."""

    is_percent: bool = False
    ulp: float = 1.0
    """Unit in the last place of the written literal: 391.0 -> 0.1, 391 -> 1."""

    text: str = ""

    @property
    def absolute(self) -> float:
        """The value with its explicit scale word applied."""
        return self.value * self.scale

    @property
    def rounding_tolerance(self) -> float:
        """
        Slack implied by how precisely the number was written.

        Only applies when a scale word was written: "391.0 billion" is a
        deliberately rounded restatement and can only be pinned down to half of
        its last digit. Bare figures such as "391,035" are exact as written.
        """
        if not self.explicit_scale:
            return 0.0
        return 0.5 * self.ulp * self.scale


@dataclass
class GradeResult:
    """Outcome of grading one answer against one golden entry."""

    passed: bool
    mode: str
    reasons: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


def normalize_text(text: str) -> str:
    """Lowercase, straighten quotes and collapse whitespace for phrase search."""
    if not text:
        return ""
    return " ".join(text.translate(_APOSTROPHES).lower().split())


def _ulp_of(digits: str) -> float:
    """Unit in the last place of a written literal ('46.25' -> 0.01)."""
    if "." in digits:
        return 10.0 ** -len(digits.split(".", 1)[1])
    return 1.0


def parse_numbers(text: str) -> List[ParsedNumber]:
    """
    Extract every numeric literal from ``text`` with its scale and precision.

    Handles comma thousands separators, currency prefixes, 'billion'/'B' and
    'million'/'M' scale words, percent signs, and accounting-style parenthesised
    negatives, where '(8)%' means -8.
    """
    if not text:
        return []

    parsed: List[ParsedNumber] = []
    for match in _NUMBER_RE.finditer(text):
        digits = match.group("digits")
        value = float(digits.replace(",", ""))

        # A negative only if the parentheses actually wrap the number; a stray
        # closing paren from surrounding prose ("(approximately $391.0 billion)")
        # must not flip the sign.
        parenthesised = bool(match.group("open")) and bool(match.group("close"))
        if parenthesised or match.group("sign"):
            value = -value

        scale_word = (match.group("scale") or "").lower()
        scale = SCALE_WORDS.get(scale_word, 1.0)

        parsed.append(
            ParsedNumber(
                value=value,
                scale=scale,
                explicit_scale=bool(scale_word),
                is_percent=bool(match.group("percent")),
                ulp=_ulp_of(digits),
                text=match.group(0).strip(),
            )
        )
    return parsed


def parse_number(text: str) -> List[float]:
    """
    Extract numeric values from a model answer, with scale words applied.

    ``"$391.0 billion"`` -> ``[391000000000.0]``, ``"391,035"`` -> ``[391035.0]``,
    ``"(8)%"`` -> ``[-8.0]``.
    """
    return [p.absolute for p in parse_numbers(text)]


def normalize_value(value: float, unit: str) -> float:
    """Convert a golden value expressed in ``unit`` to its dimension's base unit."""
    try:
        return float(value) * UNIT_SCALE[unit]
    except KeyError:
        raise ValueError(f"unknown unit: {unit!r}") from None


def _candidate_bases(parsed: ParsedNumber, unit: str) -> List[tuple]:
    """
    Interpretations of one parsed number in the base unit, as (value, slack).

    A number written with a scale word means exactly what it says. A bare number
    in a money question is ambiguous — the statements quote millions and the
    narrative quotes billions — so both readings are offered.
    """
    if unit in MAGNITUDE_UNITS:
        return [(abs(parsed.absolute), parsed.rounding_tolerance)]
    if parsed.explicit_scale:
        return [(parsed.absolute, parsed.rounding_tolerance)]
    if unit in MONEY_UNITS:
        return [(parsed.value * 1e6, 0.0), (parsed.value * 1e9, 0.0)]
    return [(parsed.value * UNIT_SCALE[unit], 0.0)]


def _number_matches(
    parsed: ParsedNumber, expected: float, unit: str, tolerance: Optional[float]
) -> bool:
    expected_base = normalize_value(expected, unit)
    tol_base = abs(normalize_value(tolerance or 0.0, unit))
    if unit in MAGNITUDE_UNITS:
        expected_base = abs(expected_base)
    slack = max(abs(expected_base), 1.0) * _FLOAT_SLACK_REL

    for candidate, extra in _candidate_bases(parsed, unit):
        if abs(candidate - expected_base) <= tol_base + extra + slack:
            return True
    return False


def _any_number_matches(
    answer: str, expected: float, unit: str, tolerance: Optional[float]
) -> bool:
    return any(
        _number_matches(p, expected, unit, tolerance) for p in parse_numbers(answer)
    )


def grade_numeric(answer: str, entry: Dict[str, Any]) -> bool:
    """
    True when the answer states the expected number(s) within tolerance.

    Any number in the answer may satisfy the constraint. When the entry declares
    a ``secondary_numeric``, that must be satisfied too. An entry with no
    ``answer_numeric`` has no numeric constraint and passes vacuously.
    """
    expected = entry.get("answer_numeric")
    if expected is None:
        return True
    if not answer or not answer.strip():
        return False

    if not _any_number_matches(
        answer, expected, entry.get("unit", "none"), entry.get("tolerance")
    ):
        return False

    secondary = entry.get("secondary_numeric")
    if secondary is None:
        return True
    return _any_number_matches(
        answer,
        secondary,
        entry.get("secondary_unit", entry.get("unit", "none")),
        entry.get("secondary_tolerance"),
    )


_CLAUSE_SPLIT_RE = re.compile(
    r"[,;:.!?—–]+"
    r"|\n+"
    r"|\s+(?:while|whereas|but|unlike|although|though|however|and|versus|vs)\s+",
    re.IGNORECASE,
)


def _split_clauses(text: str) -> List[str]:
    """
    Break text at sentence, punctuation and conjunction boundaries.

    Clause-level scoping is what lets a contrastive answer be graded correctly:
    "Greater China declined, while Americas and Europe grew" keeps the decline
    cue in a different clause from the competing segments, whereas "Europe
    declined, unlike Greater China" does not.
    """
    parts = _CLAUSE_SPLIT_RE.split(text)
    return [p for p in (part.strip() for part in parts) if p]


def _contains_any(haystack: str, needles: Sequence[str]) -> Optional[str]:
    for needle in needles:
        token = normalize_text(needle)
        if token and token in haystack:
            return needle
    return None


def grade_categorical(answer: str, entry: Dict[str, Any]) -> bool:
    """
    True when the answer names the expected entity and no competing one.

    ``expected_tokens`` (falling back to ``aliases``) supplies the entity that
    must appear. ``competing_tokens`` supplies the entities that must not be
    named as the subject of the claim.

    When the entry also declares ``competing_context``, a competing token only
    counts against the answer if it shares a *clause* with one of those context
    phrases and that clause does not also name the expected entity. That
    refinement is what lets the verified Q7 answer — "Greater China was the only
    segment with declining net sales. Americas, Europe, Japan and Rest of Asia
    Pacific all grew." — pass, while "Europe saw net sales decline, unlike
    Greater China" fails. Without ``competing_context`` the check is bare
    presence, as a plain expected-vs-competing token test.
    """
    if not answer or not answer.strip():
        return False

    normalized = normalize_text(answer)
    expected = entry.get("expected_tokens") or entry.get("aliases") or []
    if not _contains_any(normalized, expected):
        return False

    competing = entry.get("competing_tokens") or []
    if not competing:
        return True

    context = entry.get("competing_context") or []
    if not context:
        return _contains_any(normalized, competing) is None

    for clause in _split_clauses(normalized):
        if _contains_any(clause, competing) is None:
            continue
        if _contains_any(clause, context) is None:
            continue
        if _contains_any(clause, expected) is None:
            return False
    return True


def grade_keyword(answer: str, entry: Dict[str, Any]) -> bool:
    """True when every phrase in ``required_keywords`` appears in the answer."""
    required = entry.get("required_keywords") or []
    if not required:
        return True
    if not answer or not answer.strip():
        return False

    normalized = normalize_text(answer)
    return all(normalize_text(keyword) in normalized for keyword in required)


def check_direction(answer: str, entry: Dict[str, Any]) -> bool:
    """
    True when the answer's directional claim is consistent with the ledger.

    Driven entirely by the golden entry: ``forbidden_phrases`` may not appear,
    and if ``required_direction.any_of`` is set at least one of its phrases must.
    A right number with a wrong direction word therefore scores zero.
    """
    normalized = normalize_text(answer)

    forbidden = _contains_any(normalized, entry.get("forbidden_phrases") or [])
    if forbidden is not None:
        return False

    required = entry.get("required_direction") or {}
    any_of = required.get("any_of") or []
    if any_of and _contains_any(normalized, any_of) is None:
        return False
    return True


def grade(answer: str, entry: Dict[str, Any]) -> GradeResult:
    """
    Grade one answer against one golden entry.

    Runs the mode-specific check, the direction check, and — when the entry sets
    ``require_numeric`` (default: true for ``numeric`` mode) — the numeric check.
    All applicable checks must pass.
    """
    mode = entry.get("grading", "numeric")
    if mode not in ("numeric", "categorical", "keyword"):
        raise ValueError(f"unknown grading mode: {mode!r}")

    reasons: List[str] = []
    if not answer or not answer.strip():
        return GradeResult(False, mode, ["empty answer"])

    if mode == "categorical" and not grade_categorical(answer, entry):
        expected = entry.get("expected_tokens") or entry.get("aliases") or []
        reasons.append(
            f"categorical: expected one of {list(expected)}, "
            "or a competing entity was named"
        )
    if mode == "keyword" and not grade_keyword(answer, entry):
        reasons.append(f"keyword: missing {entry.get('required_keywords')}")

    if entry.get("require_numeric", mode == "numeric"):
        if not grade_numeric(answer, entry):
            expected = f"{entry.get('answer_numeric')} {entry.get('unit')}"
            if entry.get("secondary_numeric") is not None:
                expected += (
                    f" and {entry['secondary_numeric']} "
                    f"{entry.get('secondary_unit')}"
                )
            reasons.append(f"numeric: no match for {expected}")

    if not check_direction(answer, entry):
        forbidden = _contains_any(
            normalize_text(answer), entry.get("forbidden_phrases") or []
        )
        if forbidden is not None:
            reasons.append(f"direction: forbidden phrase {forbidden!r}")
        else:
            label = (entry.get("required_direction") or {}).get("label", "direction")
            reasons.append(f"direction: answer does not state '{label}'")

    return GradeResult(not reasons, mode, reasons)
