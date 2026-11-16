# Dataset Card: mintaka_hi_pool

## Summary

- **Run timestamp**: 2026-11-16T10:02:58.314461Z
- **Wall clock time**: 6.4s
- **Pipeline config hash**: `0733c89717bb3eab`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 2400 | 2400 | — | 0 | 2400 |
| SchemaGate | 2400 | 2400 | 0 | 0 | 2400 |
| ExactDeduplicator | 2400 | 2400 | — | — | 2400 |
| MinHashDeduplicator | 2400 | 2390 | — | — | 2390 |
| TextCleaner | 2390 | 2390 | — | — | 2390 |
| AlpacaExporter | — | 2390 | — | — | 2390 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 2400/0 passed (N/A), 0 rejected
- **SchemaGate**: 2400/2400 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - near_duplicates_removed: 10
  - surviving_count: 2390

## Known Limitations

- MinHash threshold used: **0.85** — this is a starting point, not a universal answer. Adjust per domain (legal, medical, and other domain-specific corpora may need a lower threshold to avoid losing structurally similar but semantically valid examples).
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.29
- **python**: 3.13.16
