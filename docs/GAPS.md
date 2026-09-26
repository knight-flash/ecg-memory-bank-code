# Remaining Verification Gaps

1. **Content substitution after raw waveform read failures.** The main HEEDB, MIMIC, and CODE-15 extraction sources were found and archived. Existing MIMIC and CODE-15 indexes, metadata, and embeddings were checked for counts and ID order; HEEDB `finalgallery` vector counts match the summaries produced with a common mask. See [the extraction source audit](EMBEDDING_SOURCE_PATHS.md). Some original dataset classes substitute the next sample when a waveform read fails, which could make a vector's contents differ from its assigned ID. Available statistics and progress files do not contain per-record failure counts. Ruling this out requires original run logs or waveform failure records. A row-by-row ID-order hash has not been computed for the very large HEEDB CSV files.

2. **No full-chain rerun on real data.** Verification to date covers syntax, import references, command dry runs, synthetic-data protocol tests, and archive file checks. The full HEEDB, MIMIC, and LLM experiments were not rerun solely to prepare this source-code archive. These checks therefore do not establish that a fresh full run will reproduce every reported number.

Plotting scripts were excluded from this archive as requested by the author. Both the low-level broad case-addition scripts and the formal entry point now enforce the final manuscript's strictly nested protocol, so this is no longer an open issue.

## Verified in this release

- **Frozen encoder checkpoint.** The original extraction configuration points to `ecgfounder_medcpt_retrieval_F_10000_total.pth`. The file exists on the server; its SHA-256 is recorded in the extraction source audit. Weights are excluded from the source-only archive.
- **Formal judge model.** All four final result summaries identify `deepseek-ai/DeepSeek-V3.1-Terminus` as `eval_llm`. The `deepseek-v3.1` string in the older execution script is the service deployment name. The formal model identity is no longer an open issue. Request preparation uses temperature 0, disables thinking, and applies a common evaluation prompt. The archived summary entry point also records the full model identity.
