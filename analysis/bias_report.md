# LLM Judge Bias Report - Phase B

**Student:** Tran Van Quang  
**Date:** 2026-06-30  
**Judge model:** gpt-4o-mini with deterministic fallback when no API key is present

## 1. Pairwise Judge Results

| # | Question ID | Winner | Reasoning summary |
|---:|---:|---|---|
| 1 | 1 | A | Model answer contains the direct policy answer. |
| 2 | 5 | B | Human note catches the CEO approval requirement. |
| 3 | 12 | A | Model answer gives the Tet bonus value directly. |
| 4 | 21 | A | Model answer includes both leave days and salary range. |
| 5 | 23 | A | Model answer gives the refund amount and commitment rule. |
| 6 | 29 | tie | The fallback judged both answers close on lexical signals. |
| 7 | 33 | A | Model answer contains both allowance and leave days. |
| 8 | 41 | B | Human note catches the v2024 vs v2023 conflict. |
| 9 | 46 | A | Model answer matches the probation leave policy. |
| 10 | 50 | B | Human note catches the personal VPN prohibition. |

## 2. Swap-And-Average Results

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---:|---|---|---|---|
| 1 | A | A | A | Yes |
| 2 | B | B | B | Yes |
| 3 | A | A | A | Yes |
| 4 | A | A | A | Yes |
| 5 | A | A | A | Yes |
| 6 | tie | tie | tie | Yes |
| 7 | A | A | A | Yes |
| 8 | B | B | B | Yes |
| 9 | A | A | A | Yes |
| 10 | B | B | B | Yes |

**Position bias rate:** 0.0% (0/10 inconsistent)

## 3. Cohen Kappa Analysis

| Question ID | Human Label | Judge Label | Agree? |
|---:|---:|---:|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 1 | Yes |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 1 | Yes |
| 29 | 0 | 0 | Yes |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 0 | Yes |

**Cohen kappa:** 1.00  
**Interpretation:** almost perfect agreement on this labeled sample.

## 4. Verbosity Bias

Among decisive cases:

- A wins and A is longer than B: 5 cases
- B wins and B is longer than A: 3 cases
- Total decisive cases: 9
- **Verbosity bias rate:** 88.9%

This high verbosity signal is expected because many correct answers in the sample are also more detailed. In production, this should be monitored with adversarial pairs where a longer answer is intentionally wrong or padded.

## 5. Overall Notes

Swap-and-average removed obvious position bias in this sample. The kappa score is strong, but the run used deterministic fallback logic rather than a live LLM judge because no API key was configured. For production, keep the deterministic path as a CI smoke test and run the actual GPT-4o-mini/OpenRouter judge for release gates and weekly audit reports.
