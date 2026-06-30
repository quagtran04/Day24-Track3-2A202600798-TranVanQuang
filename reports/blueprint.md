# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Student:** Tran Van Quang  
**Date:** 2026-06-30

## Guard Stack Pipeline

| Layer | Tool | Latency P95 | Failure Action |
|---|---|---:|---|
| PII Detection | Presidio-compatible regex fallback | 0.02ms | Reject + log |
| Topic/Jailbreak | NeMo Input-compatible rule rail | 0.05ms | 503 + reason |
| RAG Pipeline | Day 18 pipeline | <2000ms target | Fallback answer |
| Output Check | NeMo Output-compatible rule rail | <300ms target | Block/redact + log |
| Total Guard | Full guard stack | 0.07ms | Continue only if safe |

## CI Gates

- RAGAS faithfulness >= 0.75 on the 50-question evaluation set.
- RAGAS average score >= 0.65 across factual, multi_hop, and adversarial distributions.
- Adversarial suite pass rate >= 90% before merge to main.
- P95 total guard latency < 500ms.
- No `# TODO` markers in `src/phase_*.py`.
- `pytest tests/` must pass.

## Monitoring

| Metric | Current Lab Value | Alert Threshold | Action |
|---|---:|---:|---|
| RAGAS avg_score | 1.00 baseline | <0.65 | Re-run retrieval eval and inspect bottom 10 |
| Worst RAGAS metric | faithfulness tie | <0.75 | Tighten generation prompt and context grounding |
| Dominant failure distribution | factual baseline | N/A | Review distribution-level drift |
| Cohen kappa | 1.00 | <0.60 | Audit judge prompt and human-label alignment |
| Adversarial pass rate | 20/20 | <18/20 | Add attack patterns to rails |
| Guard P95 latency | 0.07ms | >500ms | Profile NeMo/API layer |

## Production Notes

This implementation supports real OpenRouter/OpenAI judging when `OPENAI_API_KEY` is set. Without a key, it uses deterministic fallback logic so CI remains runnable. In a production deployment, the fallback should be treated as a smoke-test layer, while the nightly or pre-release gate should run the true Day 18 RAG pipeline, RAGAS metrics, and LLM judge against saved evaluation sets.
