# ECG Memory Bank: paper code archive

**Release:** `v1.0.0` (final manuscript code)

This folder contains **source code and static configuration only** for the manuscript in
`../ECG_Memory_Bank_for_Governable_Use_of_Historical_Cases_in_Diagnosis_and_Interpretation.zip`.
No patient records, waveforms, embeddings, weights, generated reports, ratings, figures,
logs, or experiment outputs are included. The original `Result&Code/` directory was not changed.

## License

Original project code is released under the [MIT License](LICENSE). The adapted
ECGFounder `net1d.py` backbone also carries upstream Apache-2.0 obligations;
see [third-party notices](THIRD_PARTY_NOTICES.md) and the included license texts.
Data, model weights, and external API services are outside this code license.

## Setup

Target environment: Linux, Python 3.10.17, PyTorch 1.12.1 and NumPy 1.26.4.
`pyproject.toml` records the corresponding server package versions. Install in an
isolated environment, then run from this directory. Raw datasets and model weights
must be supplied separately under their own access terms.

```bash
python -m pip install -e .
python scripts/run_paper.py --help
python -m unittest discover -s tests -v
```

The unified runner takes a task, a JSON configuration, and a required output directory
**outside this source folder**:

```bash
python scripts/run_paper.py broad-heedb \
  --config configs/example_broad_heedb.json \
  --output-dir /path/to/experiment_outputs --dry-run
```

Replace every `/path/to/...` example before a real run. `--dry-run` prints the command
without touching the data or calling an API. `docs/DATA_CONTRACTS.md` specifies inputs;
`docs/PAPER_CODE_MAP.md` maps each manuscript analysis to its renamed code entry.

For the original dataset-specific extraction chain, use a separate external
output directory for each dataset. For example:

```bash
python scripts/run_extraction.py mimic \
  --config /path/to/retrieval_config.yaml \
  --medcpt-model /path/to/medcpt \
  --input /path/to/mimic_index.csv \
  --mimic-raw-data-dir /path/to/mimic_waveforms \
  --output-dir /path/to/experiment_outputs/mimic_extraction --dry-run
```

## Formal execution order

1. Train the frozen ECG–text encoder with `torchrun` and
   `src/ecg_memory_bank/pretraining/train_ecg_text_encoder.py`, or provide the
   verified external checkpoint. Use `scripts/run_extraction.py` for the
   original HEEDB, MIMIC and CODE-15 extraction implementations;
   `pretraining/extract_ecg_embeddings.py` is an additional adapter for
   preprocessed `.npy` waveforms.
2. Run `broad-heedb`, `broad-mimic`, `chagas-addition`, and `withdrawal` through
   `scripts/run_paper.py`. Broad addition is locked to eight nested memory states.
   For the appendix encoder comparison, run `encoder-full`, `encoder-withdrawal`,
   and `encoder-chagas` with matched ECGFounder embedding inputs.
3. Run `prototype-heedb`, `prototype-mimic`, `prototype-code15`,
   `prototype-chagas`, then `linear` for the four corresponding training tasks.
   HEEDB internal and external transfer are two evaluation settings from one head.
4. Run `report-glm46` and `report-glm5` on the same 2,381-case grounding cohort,
   then the `judge` task on each finalized report set. Run `reader-study` on the
   external blinded ratings and mapping file.

Each task writes to its own subdirectory under the supplied external output root.
The release includes statistical and audit scripts under `retrieval/`, `reports/`,
and `analysis/`; their input and output flags remain explicit. Existing experiment
results may be inspected to verify provenance, but are not part of this package.

## Important provenance notes

- The checked local retrieval and LLM scripts had identical SHA-256 hashes to the
  corresponding server copies. The HEEDB prototype base had a later local label
  mapping correction, so the corrected local version is archived.
- Pretraining code came from the server's historical ECGFounder/MedCPT source.
  The original extraction configuration and its checkpoint path/hash are recorded
  in `docs/EMBEDDING_SOURCE_PATHS.md`. Existing embedding-to-case row alignment
  still requires artifact-level verification; see `docs/GAPS.md`.
- The generated label-count table is intentionally absent. Report taxonomy code
  recomputes counts from the external metadata when no count table is supplied.
- The three dataset-specific extractors were recovered from the local `clip2026`
  workspace and matched to the server sources. Their renamed copies and the
  read-only artifact checks are listed in `docs/EMBEDDING_SOURCE_PATHS.md`.
