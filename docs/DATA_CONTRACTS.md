# Input contracts

All paths below refer to **external** files. The code archive contains no datasets,
embeddings, model weights, raw reader ratings, or generated outputs.

| Input | Required fields or form | Validation |
| --- | --- | --- |
| Record-level embeddings | PyTorch tensor `[records, dimension]`, same row order as metadata | Finite, nonzero, L2-normalized before indexing; row counts must match exactly. |
| ECG metadata | CSV with unique nonempty record ID, `labels_correct`, and experiment-specific subject/patient ID | No positional truncation. Set `record-id-col` explicitly. |
| HEEDB broad addition | Train and fixed test embeddings/metadata | Eight nested memory fractions; 114 common L3 labels for macro AUROC. |
| MIMIC broad addition | Embeddings/metadata with `subject_id` | Subject-level 80/20 split; fixed query set and 83 common L3 labels. |
| CODE-15 Chagas addition | Embeddings/metadata with record-level `chagas` indicator | Record-level 1,000-positive/4,000-negative query split; baseline has no positive reference. |
| Withdrawal | HEEDB train/val/test embeddings, metadata, acquisition year and patient ID | 2022 fixed query cohort; 15/30/60/90% patient withdrawal and 2010/2015/2018 historical withdrawal. |
| Prototype and Linear heads | Same embedding, label, and split inputs for matched setting | Prototype projection must retain a real training-case ID; 3,000 budget, 1–30 per class for the paper runs. |
| Report generation | Released 2,381-case ECG-Grounding JSON, MIMIC embedding/metadata/feature root, external classification spreadsheet | Query subjects excluded from memory; Top-5. `ZHIPU_API_KEY` is read from environment. |
| Judge | Finalized report JSONL or best-of/retry directories plus external ground truth JSONL | Shared rubric in `configs/judge_prompt.txt`; `ARK_API_KEY` or `VOLCENGINE_API_KEY` from environment. |
| Reader study | External blinded JSON ratings, A/B mapping, optional label map | Four readers × 100 cases; paired case-clustered analysis uses seed 42 and 1,000 bootstrap draws. |

The formal dataset-specific extraction entry is `scripts/run_extraction.py`;
it requires an external model config, MedCPT directory, index or raw-data root,
and output directory. The additional `.npy` adapter accepts an external array
shaped `[records, 12, samples]` and an equally ordered CSV with unique case IDs.
Both need the external full ECGCLIP checkpoint. Source and artifact paths are
listed in `docs/EMBEDDING_SOURCE_PATHS.md`.
