# EDGAR-Searcher evaluation

This directory measures two things separately: whether retrieval finds the right
part of the filing, and whether the system then produces the **right answer**.
Keeping them apart is the point — a drop in one should never be misread as a
drop in the other.

```
eval/
├── golden.json    20 verified questions about one specific filing
├── grading.py     deterministic, LLM-free grader (pure stdlib)
├── run_eval.py    the harness: retrieval metrics + end-to-end answer accuracy
└── tests/         unit tests for the grader and the retrieval matcher
```

## Which filing the golden set covers

| Field | Value |
|---|---|
| Company | Apple Inc. |
| Ticker / CIK | AAPL / 0000320193 |
| Form | 10-K |
| Fiscal year | FY2024 (52 weeks) |
| Period ended | 2024-09-28 |
| **Filed** | **2024-11-01** |
| Accession | `0000320193-24-000123` |
| Primary doc | [aapl-20240928.htm](https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm) |

Note the two different dates. The fiscal year ended 2024-09-28 and the filing
was submitted 2024-11-01. They are separate facts and the eval treats them as
separate fields — see "Retrieval metrics" below.

### How correctness was established

Every figure in `golden.json` was extracted from the primary document and then
independently re-pulled from the SEC XBRL companyfacts API
(`data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json`, filtered to
`form=10-K`, `end=2024-09-28`, `accn=0000320193-24-000123`). Both sources agree
exactly on total net sales, gross margin, operating income, net income, R&D, the
tax provision, basic/diluted EPS, total assets, total liabilities, total
shareholders' equity, cash from operations, capex, cash and equivalents,
inventories and total cost of sales.

The golden set is a **testing set**: its correctness is the whole product.
Values are transcribed verbatim. Do not recompute a stated figure and overwrite
it, and do not fill one in from memory.

## Running it

The eval talks to the same services as the app, via the same env vars:

```bash
export CHROMA_HOST=http://localhost:8000     # default
export OLLAMA_HOST=http://localhost:11434    # default
```

Load the AAPL FY2024 10-K into the collection first (app home page), then:

```bash
# retrieval + answer accuracy (needs Chroma and Ollama)
python eval/run_eval.py --collection sec_filings_embeddings_v2

# retrieval metrics only — no LLM required
python eval/run_eval.py --retrieval-only

# with a reranker, and a machine-readable dump for tracking runs over time
python eval/run_eval.py --reranker crossencoder --json eval/results_2024-11.json
```

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--golden` | `eval/golden.json` | path to the golden set |
| `--collection` | `sec_filings_embeddings_v2` | Chroma collection to query |
| `--k` | `5` | top-k cut after reranking |
| `--candidates` | `10` | initial fetch before reranking |
| `--reranker` | `off` | `off`, `bm25`, or `crossencoder` |
| `--model` | `$OLLAMA_MODEL` or `llama2` | Ollama model that answers |
| `--retrieval-only` | off | skip generation and grading |
| `--json PATH` | — | write full results as JSON |

If Chroma or Ollama is unreachable, the collection is missing or empty, or the
answering model is not pulled, the run **aborts with an explicit message**. It
never reports 0% because a service was down.

`eval/.gitignore` keeps `results_*.json` out of version control.

## What each metric means

### Retrieval metrics

| Metric | Definition |
|---|---|
| **Recall@1** | The top-ranked chunk is from the expected filing *and* one of the expected items. |
| **Recall@k** | Any of the top-k chunks is. |
| **Section-precision@k** | Of the chunks **actually returned**, the fraction sitting in one of the expected items (ticker + item only). |

Two matcher bugs the rewrite fixed:

1. **Fiscal year was matched against `filing_date[:4]`.** A filing date is not a
   fiscal year. Apple's FY2024 10-K was filed 2024-11-01 for a year ended
   2024-09-28; the FY2023 10-K was filed 2023-11-03. Filing identity is now
   matched on **accession number** first, falling back to an explicit
   fiscal-year field (`fiscal_year`, `period_end`, `period_of_report`, …) on the
   chunk metadata. It is never inferred from the filing date. A chunk carrying
   neither is reported as *unresolved* rather than being silently scored.
2. **Section-precision divided by `k`.** When fewer than `k` chunks came back,
   the score was diluted by results that were never returned. It now divides by
   `len(metas)`.

`expected_item` (a single string) also became `expected_items` (a list), because
Apple's headline figures legitimately appear in both Item 7 (MD&A) and Item 8
(financial statements).

### Answer accuracy

Each question goes through the full RAG path — retrieve, rerank, generate — and
the generated answer is graded by `grading.py`. There is **no LLM judge**:
grading is deterministic, so the same answer always scores the same way, and the
grader itself can be unit-tested. Accuracy is reported overall and broken out by
the ten coverage dimensions:

| Dimension | Questions |
|---|---|
| `revenue` | q01, q02 |
| `mix` | q03, q04, q05 |
| `segments` | q06, q07 |
| `margins` | q08, q09, q10 |
| `opex` | q11, q12 |
| `profitability` | q13, q14, q15 |
| `tax` | q16, q17 |
| `cash_flow` | q18 |
| `balance_sheet` | q19 |
| `capital_return` | q20 |

## How grading works

Three modes, set per entry by the `grading` field:

- **`numeric`** — the answer must state `answer_numeric` within `tolerance`, and
  `secondary_numeric` within `secondary_tolerance` when present. Any number in
  the answer may satisfy the constraint.
- **`categorical`** — the answer must name the expected entity
  (`expected_tokens`) and must not name a `competing_tokens` entity as the
  subject of the claim.
- **`keyword`** — every phrase in `required_keywords` must appear.

`require_numeric` lets a non-numeric mode also demand the number. Q17 uses it:
"State Aid" alone is not enough, the $10.2 billion charge must be there too.

### Units

`unit` is one of `usd_millions`, `usd_billions`, `percent`, `usd_per_share`,
`shares_millions`, `none`. Values are normalised to a common base before
comparison, so **$391.0 billion matches a golden value of 391,035 million**: a
number written with a scale word is a rounded restatement, so it carries the
tolerance implied by its own precision (±0.05 billion for "391.0 billion"). Bare
numbers are treated as exact as written; in a money question a bare number is
read as either millions or billions, since filings quote both.

Percentages are compared on **magnitude**. Natural answers write "down 8%", not
"-8%", so the sign of a percentage is graded by the direction check instead.

### Direction checks

A right number with a wrong direction word scores **zero**. This is driven
entirely by the golden entry, not by per-question logic in the grader:

- `forbidden_phrases` — if any appears, the entry fails outright.
- `required_direction.any_of` — at least one must appear.

| Question | Trap |
|---|---|
| q04 | iPhone was **flat** (em dash in the change column), not up. |
| q14 | Net income **fell** 3.4% while revenue rose 2%, because of the State Aid charge. |
| q19 | Shareholders' equity **declined** (62,146 → 56,950) as buybacks and dividends exceeded net income. |

q19 only sets `forbidden_phrases`: the question asks for two balances, not for a
direction, so a bare correct answer passes — but claiming equity rose fails.

q07 is the categorical counterpart: naming any segment other than Greater China
as the decliner fails. Competing tokens are scoped to the **clause** they appear
in, so the verified answer ("Greater China declined; Americas, Europe, Japan and
Rest of Asia Pacific all grew") passes while "Europe declined, unlike Greater
China" fails.

## Tests

```bash
python -m pytest eval/tests/ -v
```

These require **no Chroma, no Ollama and no network** — they import
`eval/grading.py` and `eval/run_eval.py` (whose backend imports are lazy) and
read `eval/golden.json` from disk. They cover number parsing, unit equivalence,
tolerance boundaries on both sides, all three grading modes, the wrong-direction
cases on q04/q14/q19, the wrong-segment case on q07, and both retrieval-matcher
fixes. One test grades every entry's own verified `answer_text` and asserts it
passes — the ledger's own wording must survive its own grader.

Style gates for this directory:

```bash
black --check eval/
flake8 --max-line-length=88 --extend-ignore=E203 eval/
```

## Regenerating or extending the golden set

`golden.json` is `{"metadata": {...}, "questions": [...]}`. Each question:

| Field | Notes |
|---|---|
| `id` | `q01`…`qNN`, unique |
| `question` | asked verbatim |
| `ticker`, `cik`, `accession`, `fiscal_year`, `period_end` | filing identity |
| `expected_items` | **list** of acceptable item numbers |
| `answer_numeric`, `unit`, `tolerance` | primary numeric answer |
| `secondary_numeric`, `secondary_unit`, `secondary_tolerance` | optional |
| `answer_text` | the verified reference answer |
| `aliases` | acceptable surface forms |
| `grading` | `numeric` \| `categorical` \| `keyword` |
| `dimension` | coverage bucket used in the breakout |
| `source` | where in the filing it comes from |
| `expected_tokens`, `competing_tokens`, `competing_context` | categorical mode |
| `required_keywords`, `require_numeric` | keyword mode |
| `required_direction`, `forbidden_phrases` | direction checks |
| `note` | why the entry is tricky |

To cover a different filing:

1. Pull the primary document from EDGAR and extract every figure you intend to
   ask about.
2. Cross-verify each one against the XBRL companyfacts API for that CIK,
   filtered to the exact accession and period end. Do not proceed on a figure
   the two sources disagree about.
3. Write the entries, setting `tolerance` to the rounding the filing itself
   uses.
4. Add `forbidden_phrases` wherever a plausible-sounding wrong direction exists,
   and `competing_tokens` wherever a plausible-sounding wrong entity exists.
   These are what stop a fluent, confidently wrong answer from scoring.
5. Extend `tests/test_grading.py` so every new trap has a test that fails
   without the fix.

Set `tolerance` from the filing's own rounding, never to make a failing answer
pass. If a figure looks wrong, stop and check the source — do not edit the
ledger to match the model.
