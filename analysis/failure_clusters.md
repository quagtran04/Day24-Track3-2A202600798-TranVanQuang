# Failure Cluster Analysis - Phase A

**Student:** Tran Van Quang  
**Date:** 2026-06-30

## 1. Aggregate RAGAS Scores By Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 1.00 | 1.00 | 1.00 |
| answer_relevancy | 1.00 | 1.00 | 1.00 |
| context_precision | 1.00 | 1.00 | 1.00 |
| context_recall | 1.00 | 1.00 | 1.00 |
| **avg_score** | **1.00** | **1.00** | **1.00** |

## 2. Bottom 10 Questions

| Rank | Distribution | Question ID | avg_score | worst_metric |
|---:|---|---:|---:|---|
| 1 | factual | 1 | 1.00 | faithfulness |
| 2 | factual | 2 | 1.00 | faithfulness |
| 3 | factual | 3 | 1.00 | faithfulness |
| 4 | factual | 4 | 1.00 | faithfulness |
| 5 | factual | 5 | 1.00 | faithfulness |
| 6 | factual | 6 | 1.00 | faithfulness |
| 7 | factual | 7 | 1.00 | faithfulness |
| 8 | factual | 8 | 1.00 | faithfulness |
| 9 | factual | 9 | 1.00 | faithfulness |
| 10 | factual | 10 | 1.00 | faithfulness |

## 3. Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 20 | 20 | 10 | 50 |
| answer_relevancy | 0 | 0 | 0 | 0 |
| context_precision | 0 | 0 | 0 | 0 |
| context_recall | 0 | 0 | 0 | 0 |

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** faithfulness

The current report is generated from a ground-truth baseline answer file because the full Day 18 dependency stack and live LLM key were not available during setup. Since every answer equals its ground truth and each context contains the same ground truth text, all metric values are 1.00. The dominant failure labels therefore come from deterministic tie-breaking in `worst_metric`, not from a real RAG failure pattern. For production evaluation, rerun `setup_answers.py` with the live pipeline and then rerun Phase A to obtain meaningful bottom-10 failures.

## 5. Suggested Fixes

| Metric weak point | Root cause | Suggested fix |
|---|---|---|
| faithfulness | Answer may not be grounded in retrieved chunks | Tighten system prompt, lower temperature, cite evidence |
| context_recall | Relevant chunks missing from retrieval | Improve chunking, add BM25, expand query rewriting |
| context_precision | Too many irrelevant chunks | Add reranking, metadata filters, version-aware retrieval |
| answer_relevancy | Answer does not match the question intent | Improve prompt template and answer format constraints |

## 6. Adversarial Distribution Notes

The baseline cannot prove whether adversarial questions are harder than factual questions because it uses ground truth as generated answers. In a real run, adversarial cases should usually score lower due to version conflicts, negation traps, and sensitive-policy ambiguity. Those cases should be tracked separately and used as a regression suite for retrieval and prompt changes.
