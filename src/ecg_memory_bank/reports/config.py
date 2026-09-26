"""Report experiment configuration.

Purpose
-------
Collect dataset paths, label taxonomy, prompt, API, and output settings for
ECG-Grounding evaluation.

Method
------
Build typed configuration objects from external environment paths and fixed
paper defaults; downstream scripts may override operational run names.

Inputs
------
External MIMIC embeddings/metadata, feature root, classification spreadsheet,
and API key from environment.

Outputs
-------
Configuration objects only; no data files or network requests are created on
import.

Paper parameters
----------------
Formal cohort is 2381 cases, retrieval K=5, prompt v2, GLM temperature=0.1,
max_tokens=8192, thinking enabled.

Shapes and meaning
------------------
One query corresponds to one ECG case; label scores are per candidate
diagnosis in [0,1], not calibrated clinical probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


ARCHIVE_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = Path(os.environ.get("ECG_OUTPUT_ROOT", "__OUTPUT_DIR_REQUIRED__"))


@dataclass
class DatasetConfig:
    train_embeddings_path: str = os.environ.get("ECG_MIMIC_EMBEDDINGS", "__MIMIC_EMBEDDINGS_REQUIRED__")
    train_metadata_path: str = os.environ.get("ECG_MIMIC_METADATA", "__MIMIC_METADATA_REQUIRED__")
    train_export_metadata_path: str = ""
    test_embeddings_path: str = os.environ.get("ECG_MIMIC_EMBEDDINGS", "__MIMIC_EMBEDDINGS_REQUIRED__")
    test_metadata_path: str = os.environ.get("ECG_MIMIC_METADATA", "__MIMIC_METADATA_REQUIRED__")
    corrected_label_metadata_path: str = os.environ.get("ECG_MIMIC_METADATA", "__MIMIC_METADATA_REQUIRED__")
    query_subset_csv: str = ""
    gallery_subset_csv: str = ""
    label_column: str = "labels_correct"
    path_column: str = "study_id"
    feature_root: str = os.environ.get("ECG_MIMIC_FEATURE_ROOT", "__MIMIC_FEATURE_ROOT_REQUIRED__")
    feature_match_key: str = "study_id"
    split_key: str = "subject_id"
    report_text_candidates: tuple[str, ...] = (
        "deid_t_diagnosis_original",
        "report",
        "text",
        "diagnosis_current",
        "diagnosis",
        "labels",
    )


@dataclass
class SamplingConfig:
    query_sample_size: int = 20
    gallery_sample_size: int = 500
    top_k: int = 5
    random_seed: int = 42


@dataclass
class ECGSerializationConfig:
    target_samples: int = 400
    decimal_places: int = 3
    lead_names: tuple[str, ...] = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    require_signal: bool = False


@dataclass
class PromptConfig:
    model: str = "glm-4-plus"
    strict_json: bool = True
    temperature: float = 0.1
    include_raw_signals: bool = False
    include_query_summary: bool = True
    include_retrieved_labels: bool = True
    include_retrieved_reports: bool = True
    prompt_version: str = "v2"


@dataclass
class LabelConfig:
    label_counts_csv: Path = Path(os.environ.get("ECG_LABEL_COUNTS_CSV", "__LABEL_COUNTS_REQUIRED__"))
    classification_xlsx: Path = Path(os.environ.get("ECG_CLASSIFICATION_XLSX", "__CLASSIFICATION_XLSX_REQUIRED__"))
    overrides_csv: Path = ARCHIVE_ROOT / "configs" / "label_mapping_overrides.csv"
    categories: tuple[str, ...] = (
        "Rhythm",
        "Conduction",
        "Ischemia events",
        "Hypertrophy",
        "Pacemaker",
        "General report",
        "ST-T changes",
        "Axis",
        "Others",
    )
    stop_labels: tuple[str, ...] = ("with", "and", "or", "(", ")", ",", ", age undetermined")


@dataclass
class BatchAPIConfig:
    api_key: str = ""
    endpoint: str = "/v4/chat/completions"
    auto_delete_input_file: bool = True
    poll_interval_seconds: int = 60
    request_dir_name: str = "requests"
    result_dir_name: str = "results"
    batch_record_name: str = "batch_jobs.json"


@dataclass
class DirectAPIConfig:
    model: str = "glm-4.6"
    endpoint: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    timeout_seconds: int = 300
    concurrency: int = 20
    batch_size: int = 20
    temperature: float = 0.1
    max_tokens: int = 8192
    stream: bool = False
    thinking_type: str = "enabled"
    backoff_seconds: tuple[int, ...] = (5, 15, 30)


@dataclass
class GroundingConfig:
    run_name: str = "ecg_grounding_full"
    query_sample_size: int = 2381
    gallery_sample_size: int = 500
    top_k: int = 5
    use_full_gallery_retrieval: bool = True
    exclude_same_patient: bool = True


@dataclass
class OutputConfig:
    run_name: str = "ecg_grounding_full"
    runs_root: Path = RUNS_ROOT

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_name


@dataclass
class ReportConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    ecg: ECGSerializationConfig = field(default_factory=ECGSerializationConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    batch: BatchAPIConfig = field(default_factory=BatchAPIConfig)
    direct: DirectAPIConfig = field(default_factory=DirectAPIConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def default_report_config() -> ReportConfig:
    return ReportConfig()
