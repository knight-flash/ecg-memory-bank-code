# Manuscript-to-code map

The authority is the actual `main.tex` and `appendix.tex` in the supplied final
manuscript ZIP. Paths below are relative to `src/ecg_memory_bank/` unless stated.

| Paper method or appendix | Formal renamed entry and support | Locked protocol or input |
| --- | --- | --- |
| Frozen Representation Model | `pretraining/train_ecg_text_encoder.py`, `scripts/run_extraction.py`, three renamed `extraction/` implementations, model and dataset helpers | ECGFounder + MedCPT; frozen, one normalized vector per eligible record; external verified checkpoint. |
| ECG Labels Mapping; Memory Construction and Vector Indexing | `retrieval/label_mapping_utils.py`, `metrics_utils.py`, `protocol.py` | Verified ID/row alignment, finite vectors, FAISS `IndexFlatIP`. |
| Retrieval and Voting Protocol; retrieval-depth appendix | `retrieval/metrics_utils.py`, `topk_robustness_sweep.py` | Unweighted multilabel vote, K=10; Union Recall at 1/3/5/10. |
| HEEDB broad case addition and log-linear appendix | `experiments/broad_case_addition_heedb.py`, `retrieval/fixed_label_auc_ci.py`, `build_support_tiers.py`, `build_heedb_loglinear_case_addition.py` | Seed 42, eight nested fractions, fixed 114-label denominator, OLS against log memory size. |
| MIMIC independent addition replication | `experiments/broad_case_addition_mimic.py`, `retrieval/fixed_label_auc_ci.py` | Subject-level 80/20 split, 159,724 fixed queries, eight nested fractions, fixed 83 labels and 1,000 paired resamples. |
| CODE-15 Chagas targeted addition | `experiments/targeted_case_addition_chagas.py` | Record-level split; 1,000 positive and 4,000 negative queries; 0/50/100/300/800/2000/5000 additions and matched negative control. |
| Patient-linked and historical withdrawal; influence locality | `experiments/active_memory_withdrawal.py`, `withdrawal_baseline_membership.py`, `withdrawal_audit_metrics.py` | 2022 holdout, 15/30/60/90% patient withdrawal, 2010/2015/2018 historical thresholds, post-removal index rebuild and Persistence@10 audit. |
| Encoder-sensitivity appendix | `encoder-full`, `encoder-withdrawal`, `encoder-chagas` in the runner, backed by `sensitivity/` scripts | ECGFounder is a sensitivity analysis, not the default encoder. Requires external matched embeddings; full-memory K=10 uses the 120-label endpoint. |
| Prototype Learning for Diagnosis and Audit; five settings | `heads/prototype_heedb_multilabel.py`, `prototype_mimic_multilabel.py`, `prototype_code15_multilabel.py`, `prototype_chagas.py`, `train_linear_head.py` | Same frozen embeddings and splits within each matched comparison; 3,000-prototype budget, 1–30 per class; projection anchors are training cases. |
| ECG-Grounding RAG/No-RAG and case-link traceability | `reports/generate_grounded_reports.py`, `generate_reports.py`, `evaluate_reports.py`, `case_link_audit.py` | All 2,381 query cases, Top-5, subject-disjoint gallery, GLM-4.6 and GLM-5 on the same prompt. |
| Retrieval-neutral reasoning judge | `reports/prepare_judge_requests.py`, `judge_reports.py`, `summarize_judge_scores.py` | One rubric and prompt, `deepseek-ai/DeepSeek-V3.1-Terminus` in final summaries, temperature 0, thinking off; server invocation uses `deepseek-v3.1` as its deployment name. |
| Human Reader Study | `analysis/reader_study.py`, `reader_study_ci.py`, `reader_agreement.py` | Four readers, 100 paired cases, seven 1–5 dimensions; case-clustered 1,000-bootstrap CI, sign test, Krippendorff α and ICC. |

| Final figure or supplement | Data-producing code |
| --- | --- |
| Figures 1–2: EMB schematic and study flow | Conceptual artwork and data-flow diagram; no numerical generation code. Final artwork is outside this source-only archive. |
| Figure 3; case-addition/withdrawal supplementary tables | Broad, Chagas, withdrawal, fixed-label CI, support-tier and log-linear entries above. |
| Figure 4; per-class and prototype-traceability tables | Four prototype heads, matched linear controls, projection/export helpers under `heads/`. |
| Figure 5; grounding, voting, judge and reader-study tables | Report, judge, case-link audit and reader-study entries above. |

The figure table maps the analyses that supply plotted values. Plotting and
manual figure-editing scripts are outside the requested code archive.

The top-level `scripts/run_paper.py` is the supported task entry. It locks the
paper's seed, retrieval depth, addition fractions, and prototype budget; per-run
config supplies only external inputs and operational settings. Auxiliary `.py`
files under each subsystem are dependencies of these formal entries.

Historical PPG experiments, patient-disjoint Chagas exploratory splits,
GLM-4.7 judge defaults, 1,000-case pilot reports, raw reader pages, and all
`outputs/` trees are excluded from the archive.
