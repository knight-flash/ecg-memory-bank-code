# ECGCLIP Embedding Extraction Source Audit

Local files under `clip2026` were checked first, then matched to server files by SHA-256. The paths below identify source code and configuration only; patient data and model weights were not copied into this code archive.

| Dataset / purpose | Local source (relative to `clip2026`) | Corresponding server source | Finding |
| --- | --- | --- | --- |
| HEEDB main extraction | `ECGCLIP/ecgfounder2/extract_heedb_embeddings_distributed_robust.py` | `/home/yanmingke/ecgfounder2/extract_heedb_embeddings_distributed_robust.py` | Extraction logic matches. Only default index and output paths differ: the local copy points to training data, while the current server copy points to validation data. |
| MIMIC-IV-ECG main extraction | `ECGCLIP/ecgfounder2/downstream/Preprocessing/feature_extract_mimic.py` | `/home/yanmingke/ecgfounder2/downstream/Preprocessing/feature_extract_mimic.py` | SHA-256 matches exactly. |
| CODE-15 main extraction | `ecgfounder2_extraction/extract_code15_ecgclip_single_gpu_robust.py` | `/home/yanmingke/ecgfounder2/ecgfounder2_extraction/extract_code15_ecgclip_single_gpu_robust.py` | SHA-256 matches exactly. |

The three main extraction sources were renamed in this archive to `extraction/extract_heedb_embeddings.py`, `extract_mimic_embeddings.py`, and `extract_code15_embeddings.py`. The common entry point is `scripts/run_extraction.py`. The archive contains source code only; generated embeddings and intermediate checkpoints go to external output directories.

Shared model configuration:

- Local: `ECGCLIP/ecgfounder2/retrieval/retrieval_config.yaml`
- Server: `/home/yanmingke/ecgfounder2/retrieval/retrieval_config.yaml`
- Both configurations have SHA-256 `bfd7b95601fd32b202f86ef004bb793bd3b42321a9666034fdcfd953f8034e4f`. They specify `model_name: ECGCLIP`, `ecg_model: ecgfounder`, and `text_model: medcpt`.
- The server checkpoint referenced by the configuration is `/data2/2shared/yanmingke/ecg_model/retrieval_model/ecgfounder_medcpt_retrieval_F_10000_total.pth`. It exists, is approximately 777 MiB, and has SHA-256 `d8e387edd8e74d177492526a8990caa094e8862de2dd237a130ca351c5ecbec3`.

These paths show that all three main extraction chains **default to** the same ECGCLIP model configuration and checkpoint. If an actual run overrode `--config_path`, its run record takes precedence. The two default-path differences between the local and server HEEDB scripts do not change checkpoint loading or waveform extraction functions.

**Read-only check of existing MIMIC files:** the original index `mimic/mimic_final_with_labels.csv`, extraction-time `feature/mimic/metadata.csv`, paper-use `feature/mimic/metadata_cleaned_with_labels_correct.csv`, and `feature/mimic/ecg_embeddings.pt` each contain **800,034 rows or vectors**. The `study_id` order matches row by row across all three CSV files. The original index has **zero** missing `path` or `final_report` entries, so the extraction dataset's `dropna` removed no rows. This supports positional alignment of MIMIC metadata and embeddings. On a waveform read failure, however, the script may recursively substitute the next record; excluding such rare substitutions requires the original run log or a fresh audit of the raw waveforms.

**Read-only check of existing CODE-15 files:** the original index, saved metadata, and embeddings each contain **342,755 records**. The two CSV files have matching `exam_id` order row by row, and metadata IDs are unique.

**HEEDB `finalgallery` check:** train, validation, and test vector shapes are **(8,182,498, 256)**, **(1,041,322, 256)**, and **(1,041,324, 256)**. Their first dimensions match the retained counts in `finalgallery_summary.json`. At write time, `build_finalgallery_nonempty.py` applies the same Boolean mask to metadata and original vectors after checking that their original row counts match. A row-by-row ID-order hash was not computed for the existing multi-gigabyte CSV files.

These checks support within-dataset row-order consistency. When individual raw waveforms cannot be read, some original dataset classes substitute the next sample. To rule out content substitution, inspect the corresponding run logs or per-record waveform error records.

The available HEEDB validation and test `extraction_stats.txt` files record the original model configuration; the test statistics explicitly record the checkpoint path above. The CODE-15 `progress.json` reports `completed`, but its summary has no per-record failure or substitution count. No corresponding failure-statistics log was found in the MIMIC output directory.
