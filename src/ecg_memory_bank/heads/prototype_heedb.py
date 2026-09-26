#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traditional ProtoPNet-style baseline for HEEDB training with MIMIC external test.

This script is a standalone implementation for the HEEDB -> MIMIC protocol:

1. Train / validate / internal test on HEEDB embeddings
2. External test on full MIMIC embeddings
3. Keep the traditional baseline features:
   - fixed per-class prototype allocation
   - centroid / KMeans initialization
   - periodic projection to real training samples
   - last-layer convex optimization
   - prototype mapping report
   - activation frequency report
   - t-SNE visualization
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
for candidate in (BASE_DIR, PROJECT_ROOT / "common"):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.append(str(candidate))

from ecg_memory_bank.heads.label_mapping_utils import normalize_label_text, parse_label_field
from json_export_utils import DEFAULT_PATH_FIELDS
from training_json_export_bridge import build_prototype_candidate_bank, export_embedding_showcase

warnings.filterwarnings("ignore")

EXPERIMENT_NAME = "ape_traditional"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("ECG_OUTPUT_ROOT", "__OUTPUT_DIR_REQUIRED__")) / "protopnet" / EXPERIMENT_NAME

DATA_ECG_PATH = "__ECG_DATA_ROOT_REQUIRED__/heedb_embedding_processed/2stage_experiment/stage1_dedup_ecg_embeddings.pt"
DATA_META_PATH = "__ECG_DATA_ROOT_REQUIRED__/heedb_embedding_processed/2stage_experiment/stage1_dedup_metadata_with_labels_correct.csv"
DATA_COL = "labels_correct"
HEEDB_TRAIN_EXPORT_META_PATH = "__ECG_DATA_ROOT_REQUIRED__/heedb_embedding_final/train_processed_update.csv"
VAL_ECG_PATH = "__ECG_DATA_ROOT_REQUIRED__/heedb_embedding_val/heedb_ecg_embeddings.pt"
VAL_META_PATH = "__ECG_DATA_ROOT_REQUIRED__/heedb_embedding_val/heedb_texts_with_labels_correct.csv"
VAL_COL = "labels_correct"

TEST_ROBUST_ECG_PATH = "__ECG_DATA_ROOT_REQUIRED__/heedb_embedding_test/heedb_ecg_embeddings.pt"
TEST_ROBUST_META_PATH = "__ECG_DATA_ROOT_REQUIRED__/heedb_embedding_test/heedb_metadata_with_labels_correct.csv"
TEST_ROBUST_COL = "labels_correct"
HEEDB_TEST_EXPORT_META_PATH = TEST_ROBUST_META_PATH

TEST_MIMIC_ECG_PATH = "__ECG_DATA_ROOT_REQUIRED__/feature/mimic/ecg_embeddings.pt"
TEST_MIMIC_META_PATH = "__ECG_DATA_ROOT_REQUIRED__/feature/mimic/metadata_cleaned_with_labels_correct.csv"
TEST_MIMIC_COL = "labels_correct"

DEFAULT_PROTOTYPES_PER_CLASS = 3
PROTOTYPE_COUNT_OVERRIDES = {
    "Atrial fibrillation/flutter": 6,
    "Premature ventricular complex": 6,
    "Premature atrial complex": 6,
    "Right bundle branch block": 4,
    "Left bundle branch block": 4,
    "Left ventricular hypertrophy": 5,
    "ST-T change": 10,
    "Low QRS voltage": 5,
    "Normal ECG": 6,
}

TRAIN_BATCH_SIZE = 8192
EVAL_BATCH_SIZE = 16384
INIT_MAX_SAMPLES_PER_CLASS = 50000
MAX_EPOCHS = 20
EVAL_INTERVAL = 2
PROJECTION_INTERVAL = 5
LAST_LAYER_FINE_TUNE_EPOCHS = 3
GLOBAL_LR = 1e-3
LAST_LAYER_LR = 2e-3
CLASSIFICATION_WEIGHT = 1.0
CLUSTER_WEIGHT = 0.8
SEPARATION_WEIGHT = 0.08
SEPARATION_MARGIN = 1.0
L1_WEIGHT = 1e-4
RANDOM_SEED = 42
TSNE_SAMPLE_SIZE = 4000

OUTPUT_ROOT_DIR = str(DEFAULT_OUTPUT_ROOT)
AUTO_ROOT_DIR = OUTPUT_ROOT_DIR
CHECKPOINT_ROOT_DIR = os.path.join(OUTPUT_ROOT_DIR, "checkpoint")

BEST_MODEL_PATH = os.path.join(CHECKPOINT_ROOT_DIR, "ape_traditional_protopnet_best.pt")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_ROOT_DIR, "ape_traditional_protopnet_last.pt")
BEST_SNAPSHOT_PATH = os.path.join(CHECKPOINT_ROOT_DIR, "ape_traditional_protopnet_best_snapshot.pt")
PROTOTYPE_REPORT_CSV = os.path.join(OUTPUT_ROOT_DIR, "ape_traditional_protopnet_prototype_report.csv")
HEEDB_ACTIVATION_CSV = os.path.join(OUTPUT_ROOT_DIR, "ape_traditional_protopnet_heedb_activation_report.csv")
MIMIC_ACTIVATION_CSV = os.path.join(OUTPUT_ROOT_DIR, "ape_traditional_protopnet_mimic_activation_report.csv")
PROTOTYPE_TSNE_PNG = os.path.join(OUTPUT_ROOT_DIR, "ape_traditional_protopnet_tsne.png")
JSON_EXPORT_DIR_HEEDB = os.path.join(OUTPUT_ROOT_DIR, "ape_traditional_protopnet_json_heedb")
JSON_EXPORT_DIR_MIMIC = os.path.join(OUTPUT_ROOT_DIR, "ape_traditional_protopnet_json_mimic")
JSON_EXPORT_QUERY_COUNT = 12
JSON_EXPORT_QUERY_PER_CLASS = 2
JSON_EXPORT_TOP_K = 8

CLASSIFICATION_REFERENCE_CSV = BASE_DIR / "diagnoses_text_label_counts.csv"
LABEL_PARSE_SEPARATORS = ("|", ";", "\n")
NORMAL_LABEL = "Normal ECG"
NORMAL_NEGATIVE_HINTS = (
    "abnormal ecg",
    "borderline ecg",
    "possible",
    "cannot rule out",
    "poor data quality",
)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traditional ProtoPNet-style HEEDB training with HEEDB internal and MIMIC external evaluation.",
    )
    parser.add_argument("--data-ecg-path", default=DATA_ECG_PATH)
    parser.add_argument("--data-meta-path", default=DATA_META_PATH)
    parser.add_argument("--data-label-col", default=DATA_COL)
    parser.add_argument("--heedb-train-export-meta-path", default=HEEDB_TRAIN_EXPORT_META_PATH)
    parser.add_argument("--val-ecg-path", default=VAL_ECG_PATH)
    parser.add_argument("--val-meta-path", default=VAL_META_PATH)
    parser.add_argument("--val-label-col", default=VAL_COL)
    parser.add_argument("--test-robust-ecg-path", default=TEST_ROBUST_ECG_PATH)
    parser.add_argument("--test-robust-meta-path", default=TEST_ROBUST_META_PATH)
    parser.add_argument("--test-robust-label-col", default=TEST_ROBUST_COL)
    parser.add_argument("--heedb-test-export-meta-path", default=HEEDB_TEST_EXPORT_META_PATH)
    parser.add_argument("--mimic-ecg-path", default=TEST_MIMIC_ECG_PATH)
    parser.add_argument("--mimic-meta-path", default=TEST_MIMIC_META_PATH)
    parser.add_argument("--mimic-label-col", default=TEST_MIMIC_COL)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def apply_runtime_config(args: argparse.Namespace) -> None:
    global DATA_ECG_PATH, DATA_META_PATH, DATA_COL
    global HEEDB_TRAIN_EXPORT_META_PATH
    global VAL_ECG_PATH, VAL_META_PATH, VAL_COL
    global TEST_ROBUST_ECG_PATH, TEST_ROBUST_META_PATH, TEST_ROBUST_COL, HEEDB_TEST_EXPORT_META_PATH
    global TEST_MIMIC_ECG_PATH, TEST_MIMIC_META_PATH, TEST_MIMIC_COL
    global OUTPUT_ROOT_DIR, AUTO_ROOT_DIR, CHECKPOINT_ROOT_DIR
    global BEST_MODEL_PATH, LAST_MODEL_PATH, BEST_SNAPSHOT_PATH
    global PROTOTYPE_REPORT_CSV, HEEDB_ACTIVATION_CSV, MIMIC_ACTIVATION_CSV
    global PROTOTYPE_TSNE_PNG, JSON_EXPORT_DIR_HEEDB, JSON_EXPORT_DIR_MIMIC
    global RANDOM_SEED

    DATA_ECG_PATH = args.data_ecg_path
    DATA_META_PATH = args.data_meta_path
    DATA_COL = args.data_label_col
    HEEDB_TRAIN_EXPORT_META_PATH = args.heedb_train_export_meta_path
    VAL_ECG_PATH = args.val_ecg_path
    VAL_META_PATH = args.val_meta_path
    VAL_COL = args.val_label_col
    TEST_ROBUST_ECG_PATH = args.test_robust_ecg_path
    TEST_ROBUST_META_PATH = args.test_robust_meta_path
    TEST_ROBUST_COL = args.test_robust_label_col
    HEEDB_TEST_EXPORT_META_PATH = args.heedb_test_export_meta_path
    TEST_MIMIC_ECG_PATH = args.mimic_ecg_path
    TEST_MIMIC_META_PATH = args.mimic_meta_path
    TEST_MIMIC_COL = args.mimic_label_col
    RANDOM_SEED = int(args.seed)

    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else output_root / "checkpoint"
    OUTPUT_ROOT_DIR = str(output_root)
    AUTO_ROOT_DIR = OUTPUT_ROOT_DIR
    CHECKPOINT_ROOT_DIR = str(checkpoint_root)

    BEST_MODEL_PATH = str(checkpoint_root / "ape_traditional_protopnet_best.pt")
    LAST_MODEL_PATH = str(checkpoint_root / "ape_traditional_protopnet_last.pt")
    BEST_SNAPSHOT_PATH = str(checkpoint_root / "ape_traditional_protopnet_best_snapshot.pt")
    PROTOTYPE_REPORT_CSV = str(output_root / "ape_traditional_protopnet_prototype_report.csv")
    HEEDB_ACTIVATION_CSV = str(output_root / "ape_traditional_protopnet_heedb_activation_report.csv")
    MIMIC_ACTIVATION_CSV = str(output_root / "ape_traditional_protopnet_mimic_activation_report.csv")
    PROTOTYPE_TSNE_PNG = str(output_root / "ape_traditional_protopnet_tsne.png")
    JSON_EXPORT_DIR_HEEDB = str(output_root / "ape_traditional_protopnet_json_heedb")
    JSON_EXPORT_DIR_MIMIC = str(output_root / "ape_traditional_protopnet_json_mimic")

# Refreshed from the corrected labels workflow and the observed tag phrases in
# diagnoses_text_label_counts.csv so the training labels follow the updated
# metadata instead of the older loose text heuristics.
PAPER_18_CLASS_CONFIG = {
    "Atrial fibrillation/flutter": {
        "prompt": "An ECG recording showing atrial fibrillation or atrial flutter, including variable ventricular response patterns.",
        "keywords": [
            "atrial fibrillation/flutter",
            "atrial fibrillation",
            "atrial flutter",
            "afib",
            "a-fib",
            "aflutter",
            "with rapid ventricular response",
            "with slow ventricular response",
            "with variable av block",
        ],
    },
    "Supraventricular tachycardia": {
        "prompt": "An ECG recording showing supraventricular tachycardia or another narrow-complex supraventricular tachyarrhythmia.",
        "keywords": [
            "supraventricular tachycardia",
            "svt",
            "psvt",
            "atrial tachycardia",
            "av nodal reentrant tachycardia",
            "avnrt",
            "avrt",
            "ectopic atrial tachycardia",
            "junctional tachycardia",
        ],
    },
    "Sinus tachycardia": {
        "prompt": "An ECG recording showing sinus tachycardia.",
        "keywords": [
            "sinus tachycardia",
            "sinus tach",
        ],
    },
    "Sinus bradycardia": {
        "prompt": "An ECG recording showing sinus bradycardia.",
        "keywords": [
            "sinus bradycardia",
            "sinus brady",
            "marked sinus bradycardia",
        ],
    },
    "Premature ventricular complex": {
        "prompt": "An ECG recording showing premature ventricular complexes or frequent ventricular ectopy.",
        "keywords": [
            "premature ventricular complex",
            "premature ventricular complexes",
            "pvc",
            "ventricular premature",
            "vpc",
            "ventricular ectopics",
            "ventricular bigeminy",
            "ventricular trigeminy",
            "with premature ventricular or aberrantly conducted complexes",
            "with occasional",
            "with frequent",
            "in a pattern of bigeminy",
            "fusion complexes",
        ],
    },
    "Premature atrial complex": {
        "prompt": "An ECG recording showing premature atrial complexes or supraventricular ectopy.",
        "keywords": [
            "premature atrial complex",
            "premature atrial complexes",
            "pac",
            "atrial premature",
            "apc",
            "atrial ectopics",
            "atrial bigeminy",
        ],
    },
    "Pacemaker rhythm": {
        "prompt": "An ECG recording showing a paced rhythm from an implanted pacemaker.",
        "keywords": [
            "pacemaker",
            "paced rhythm",
            "pacing",
            "atrial paced",
            "ventricular paced",
            "dual paced",
            "ventricular-paced rhythm",
            "ventricular-paced complexes",
            "atrial-paced rhythm",
            "atrial-paced complexes",
            "atrial-sensed ventricular-paced rhythm",
            "av dual-paced rhythm",
            "av dual-paced complexes",
        ],
    },
    "Atrioventricular block": {
        "prompt": "An ECG recording showing second-degree or higher atrioventricular block.",
        "keywords": [
            "atrioventricular block",
            "av block",
            "avb",
            "heart block",
            "second degree",
            "third degree",
            "complete heart block",
            "mobitz",
            "wenckebach",
            "with 2nd degree av block",
            "with 2nd degree av block (mobitz i)",
            "with 2nd degree av block (mobitz ii)",
            "with complete heart block",
            "with 2:1 av conduction",
            "with 3:1 av conduction",
            "with 4:1 av conduction",
            "with 5:1 av conduction",
        ],
    },
    "1st degree atrioventricular block": {
        "prompt": "An ECG recording showing first-degree atrioventricular block or prolonged AV conduction.",
        "keywords": [
            "1st degree atrioventricular block",
            "with 1st degree av block",
            "1st degree av block",
            "first degree av block",
            "1st degree heart block",
            "first degree heart block",
            "prolonged pr interval",
            "1-avb",
            "grade 1 av block",
            "prolonged av conduction",
            "with prolonged av conduction",
        ],
    },
    "Right bundle branch block": {
        "prompt": "An ECG recording showing right bundle branch block.",
        "keywords": [
            "right bundle branch block",
            "rbbb",
            "incomplete right bundle branch block",
            "incomplete rbbb",
            "complete rbbb",
            "rsr' or qr pattern in v1 suggests right ventricular conduction delay",
        ],
    },
    "Left bundle branch block": {
        "prompt": "An ECG recording showing left bundle branch block.",
        "keywords": [
            "left bundle branch block",
            "lbbb",
            "incomplete left bundle branch block",
            "incomplete lbbb",
            "complete lbbb",
        ],
    },
    "Left ventricular hypertrophy": {
        "prompt": "An ECG recording showing left ventricular hypertrophy or voltage criteria consistent with LVH.",
        "keywords": [
            "left ventricular hypertrophy",
            "lvh",
            "left ventricular enlargement",
            "minimal voltage criteria for lvh",
            "moderate voltage criteria for lvh",
            "voltage criteria for left ventricular hypertrophy",
            "cornell product",
            "r in avl",
            "sokolow-lyon",
            "romhilt-estes",
            "with strain pattern",
            "with repolarization abnormality",
        ],
    },
    "ST-T change": {
        "prompt": "An ECG recording showing ST-segment or T-wave abnormalities, including ischemic, infarct-related, or repolarization changes.",
        "keywords": [
            "st-t change",
            "st-t abnormality",
            "st-t wave changes",
            "nonspecific t wave abnormality",
            "nonspecific st and t wave abnormality",
            "nonspecific st abnormality",
            "t wave abnormality",
            "t wave inversion",
            "st depression",
            "st elevation",
            "repolarization abnormality",
            "early repolarization",
            "ischemia",
            "injury",
            "infarct",
            "acute mi",
            "stemi",
            "abnormal qrs-t angle",
            "acute pericarditis",
        ],
    },
    "Prolonged QT interval": {
        "prompt": "An ECG recording showing prolonged QT interval.",
        "keywords": [
            "prolonged qt interval",
            "prolonged qt",
            "long qt",
            "qt prolongation",
            "qt interval prolonged",
            "qtcb >=",
            "qtcf >=",
        ],
    },
    "Low QRS voltage": {
        "prompt": "An ECG recording showing low QRS voltage.",
        "keywords": [
            "low qrs voltage",
            "low voltage qrs",
            "low voltage",
            "low amplitude qrs",
            "diminished qrs",
        ],
    },
    "Borderline ECG": {
        "prompt": "An ECG recording described as borderline ECG.",
        "keywords": [
            "borderline ecg",
            "borderline electrocardiogram",
        ],
    },
    "Abnormal ECG": {
        "prompt": "An ECG recording described as abnormal ECG.",
        "keywords": [
            "abnormal ecg",
            "abnormal electrocardiogram",
        ],
    },
    "Normal ECG": {
        "prompt": "An ECG recording showing a normal ECG or normal sinus rhythm without major abnormalities.",
        "keywords": [
            "normal ecg",
            "normal electrocardiogram",
            "within normal limits",
            "otherwise normal ecg",
            "normal sinus rhythm",
            "sinus rhythm",
        ],
    },
}

PAPER_18_CLASSES = {
    label_name: list(config["keywords"])
    for label_name, config in PAPER_18_CLASS_CONFIG.items()
}
EXTENDED_TARGET_ALIASES = {
    "Premature ventricular complex": [
        "premature ventricular complexes",
        "with premature ventricular or aberrantly conducted complexes",
        "fusion complexes",
        "in a pattern of bigeminy",
    ],
    "Premature atrial complex": [],
    "Pacemaker rhythm": [
        "ventricular-paced rhythm",
        "atrial-paced rhythm",
        "atrial-sensed ventricular-paced rhythm",
        "av dual-paced rhythm",
        "av sequential or dual chamber electronic pacemaker",
        "ventricular-paced complexes",
        "atrials-paced complexes",
        "av dual-paced complexes",
        "electronic atrial pacemaker",
        "electronic ventricular pacemaker",
        "biventricular pacemaker detected",
        "suspect unspecified pacemaker failure",
    ],
    "Atrioventricular block": [
        "with 2nd degree av block mobitz i",
        "with complete heart block",
        "with variable av block",
        "with 2:1 av conduction",
        "with 2nd degree av conduction",
        "with 3:1 av conduction",
        "with 4:1 av conduction",
        "with av dissociation",
        "with retrograde conduction",
    ],
    "1st degree atrioventricular block": [
        "with 1st degree av block",
        "with prolonged av conduction",
    ],
    "ST-T change": [
        "nonspecific t wave abnormality",
        "nonspecific st and t wave abnormality",
        "nonspecific st abnormality",
        "with repolarization abnormality",
        "early repolarization",
        "t wave abnormality, consider lateral ischemia",
        "t wave abnormality, consider inferior ischemia",
        "t wave abnormality, consider anterior ischemia",
        "t wave abnormality, consider anterolateral ischemia",
        "t wave abnormality, consider inferolateral ischemia",
        "marked t wave abnormality, consider anterolateral ischemia",
        "marked t wave abnormality, consider lateral ischemia",
        "marked t wave abnormality, consider inferolateral ischemia",
        "marked t wave abnormality, consider inferior ischemia",
        "marked t wave abnormality, consider anterior ischemia",
        "t wave abnormality, consider septal ischemia",
        "st depression, consider anterolateral ischemia",
    ],
    "Low QRS voltage": [
        "low voltage qrs",
    ],
}
TARGET_LABELS = list(PAPER_18_CLASSES.keys())
TARGET_KEYWORDS = [PAPER_18_CLASSES[label_name] for label_name in TARGET_LABELS]
TARGET_PROMPTS = [PAPER_18_CLASS_CONFIG[label_name]["prompt"] for label_name in TARGET_LABELS]
TARGET_ALIAS_SETS = [
    {
        normalize_label_text(alias)
        for alias in (
            list(PAPER_18_CLASSES[label_name]) + EXTENDED_TARGET_ALIASES.get(label_name, [])
        )
    }
    for label_name in TARGET_LABELS
]
NORMAL_LABEL = "Normal ECG"
NORMAL_CLASS_IDX = TARGET_LABELS.index(NORMAL_LABEL)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_standardize(emb_path, meta_path, col_name):
    print(f"--> Loading Dataset: {emb_path} ...")
    ecg_emb = torch.load(emb_path, map_location="cpu")
    ecg_emb = F.normalize(ecg_emb, p=2, dim=1)
    ecg_emb = torch.nan_to_num(ecg_emb, nan=0.0)

    df = pd.read_csv(meta_path)
    if col_name not in df.columns:
        fallback_candidates = ["labels_correct", "disease_labels", "labels"]
        for fallback_col in fallback_candidates:
            if fallback_col in df.columns:
                print(f"Warning: Column '{col_name}' not found. Using '{fallback_col}' instead.")
                col_name = fallback_col
                break
        else:
            raise KeyError(f"No valid label column found in {meta_path}. Tried: {[col_name] + fallback_candidates}")
    if len(df) != len(ecg_emb):
        min_len = min(len(df), len(ecg_emb))
        print(f"Warning: Metadata len {len(df)} != Emb len {len(ecg_emb)}")
        ecg_emb = ecg_emb[:min_len]
        df = df.iloc[:min_len]

    raw_labels = df[col_name].fillna("").astype(str).tolist()
    return ecg_emb, raw_labels, df.reset_index(drop=True)


def attach_export_metadata(base_df: pd.DataFrame, export_meta_path: str) -> pd.DataFrame:
    if not os.path.exists(export_meta_path):
        print(f"[JSON Export] Warning: export metadata not found: {export_meta_path}")
        return base_df.reset_index(drop=True)

    export_df = pd.read_csv(export_meta_path)
    if "HashFileName" not in export_df.columns:
        print(f"[JSON Export] Warning: HashFileName missing in {export_meta_path}")
        return base_df.reset_index(drop=True)

    export_df = export_df.drop_duplicates(subset=["HashFileName"])
    merged_df = base_df.merge(
        export_df,
        on="HashFileName",
        how="left",
        suffixes=("", "_export"),
    )

    for field_name in DEFAULT_PATH_FIELDS:
        export_field_name = f"{field_name}_export"
        if export_field_name not in merged_df.columns:
            continue

        if field_name not in merged_df.columns:
            merged_df[field_name] = merged_df[export_field_name]
            continue

        primary_series = merged_df[field_name]
        fallback_series = merged_df[export_field_name]
        primary_missing = primary_series.isna() | primary_series.astype(str).str.strip().str.lower().isin({"", "nan", "none"})
        merged_df.loc[primary_missing, field_name] = fallback_series.loc[primary_missing]

    return merged_df.reset_index(drop=True)


def build_mimic_comparable_split(x_mimic: torch.Tensor, y_mimic: torch.Tensor, mimic_df: pd.DataFrame) -> dict[str, object]:
    indices = np.arange(len(x_mimic))
    train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=RANDOM_SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=RANDOM_SEED)
    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "x_test": x_mimic[test_idx].float(),
        "y_test": y_mimic[test_idx].float(),
        "df_test": mimic_df.iloc[test_idx].reset_index(drop=True),
    }


def has_exportable_waveform_paths(df: pd.DataFrame) -> bool:
    if df.empty:
        return False

    for field_name in DEFAULT_PATH_FIELDS:
        if field_name not in df.columns:
            continue
        series = df[field_name]
        valid_mask = series.notna() & ~series.astype(str).str.strip().str.lower().isin({"", "nan", "none"})
        if bool(valid_mask.any()):
            return True
    return False

def build_ground_truth_matrix(raw_labels):
    num_samples = len(raw_labels)
    num_classes = len(TARGET_LABELS)
    gt_matrix = torch.zeros((num_samples, num_classes))

    for i, raw_label_text in enumerate(raw_labels):
        parsed_labels = set(parse_label_field(raw_label_text))
        positive_indices = set()

        for class_idx, alias_set in enumerate(TARGET_ALIAS_SETS):
            if parsed_labels.intersection(alias_set):
                positive_indices.add(class_idx)

        for class_idx in positive_indices:
            gt_matrix[i, class_idx] = 1.0

    return gt_matrix


def print_classification_prompt_plan():
    print("\n=== Classification Prompt Plan ===")
    print(f"Reference tags: {CLASSIFICATION_REFERENCE_CSV}")
    for label_name, prompt in zip(TARGET_LABELS, TARGET_PROMPTS):
        print(f"{label_name:<35} | {prompt}")


def evaluate_performance(y_true, y_logits, title="Evaluation"):
    y_scores = torch.sigmoid(y_logits).detach().cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    print(f"\n=== {title} ===")
    print(f"{'Class':<35} | {'AUROC':<10} | {'Count'}")
    print("-" * 65)

    aucs = []
    class_auc_dict = {}

    for i, name in enumerate(TARGET_LABELS):
        count = np.sum(y_true_np[:, i])
        if count > 5 and count < len(y_true_np) - 5:
            try:
                auc = roc_auc_score(y_true_np[:, i], y_scores[:, i])
                aucs.append(auc)
                class_auc_dict[i] = auc
                print(f"{name:<35} | {auc:.4f}     | {int(count)}")
            except Exception:
                print(f"{name:<35} | Error      | {int(count)}")
        else:
            print(f"{name:<35} | N/A        | {int(count)}")

    mean_auc = np.mean(aucs) if aucs else 0.0
    print("-" * 65)
    print(f"[{title}] Mean AUROC: {mean_auc:.4f}")
    return mean_auc, class_auc_dict


def build_prototype_count_plan():
    return [PROTOTYPE_COUNT_OVERRIDES.get(label_name, DEFAULT_PROTOTYPES_PER_CLASS) for label_name in TARGET_LABELS]


def print_prototype_count_plan(counts):
    print("\n=== Prototype Count Plan ===")
    total = 0
    for label_name, count in zip(TARGET_LABELS, counts):
        total += count
        print(f"{label_name:<35} | {count}")
    print("-" * 55)
    print(f"{'Total prototypes':<35} | {total}")


def initialize_fixed_prototypes(x_train, y_train, counts, max_samples_per_class):
    print("\n--> Initializing fixed class-specific prototypes ...")
    prototype_vectors = []
    prototype_labels = []
    feature_dim = x_train.shape[1]

    for class_idx, num_prototypes in enumerate(counts):
        positive_mask = y_train[:, class_idx] == 1.0
        positive_samples = x_train[positive_mask]
        label_name = TARGET_LABELS[class_idx]

        if positive_samples.shape[0] == 0:
            print(f"  [Warning] {label_name}: no positive samples. Using random unit vectors.")
            random_protos = torch.randn(num_prototypes, feature_dim)
            random_protos = F.normalize(random_protos, p=2, dim=1)
            prototype_vectors.append(random_protos)
            prototype_labels.extend([class_idx] * num_prototypes)
            continue

        candidate_pool = positive_samples
        if candidate_pool.shape[0] > max_samples_per_class:
            perm = torch.randperm(candidate_pool.shape[0])[:max_samples_per_class]
            candidate_pool = candidate_pool[perm]

        if num_prototypes == 1:
            class_prototypes = F.normalize(candidate_pool.mean(dim=0, keepdim=True), p=2, dim=1)
        else:
            num_clusters = min(num_prototypes, candidate_pool.shape[0])
            if num_clusters > 1:
                try:
                    kmeans = KMeans(n_clusters=num_clusters, n_init=3, random_state=RANDOM_SEED)
                    kmeans.fit(candidate_pool.cpu().numpy())
                    class_prototypes = torch.tensor(kmeans.cluster_centers_, dtype=x_train.dtype)
                except Exception:
                    centers = []
                    for _ in range(num_clusters):
                        sample_idx = torch.randint(0, candidate_pool.shape[0], (1,)).item()
                        centers.append(candidate_pool[sample_idx].clone().unsqueeze(0))
                    class_prototypes = torch.cat(centers, dim=0)
            else:
                class_prototypes = candidate_pool[:1].clone()

            if num_clusters < num_prototypes:
                centroid = candidate_pool.mean(dim=0, keepdim=True)
                extras = centroid.repeat(num_prototypes - num_clusters, 1)
                extras = extras + 0.01 * torch.randn_like(extras)
                class_prototypes = torch.cat([class_prototypes, extras], dim=0)

            class_prototypes = F.normalize(class_prototypes, p=2, dim=1)

        prototype_vectors.append(class_prototypes)
        prototype_labels.extend([class_idx] * num_prototypes)
        print(f"  {label_name:<35} | positives={positive_samples.shape[0]:<8} | prototypes={num_prototypes}")

    return torch.cat(prototype_vectors, dim=0), torch.tensor(prototype_labels, dtype=torch.long)


class TraditionalProtoPNet(nn.Module):
    def __init__(self, init_prototypes, prototype_labels, num_classes):
        super().__init__()
        feature_dim = init_prototypes.shape[1]

        self.adapter = nn.Linear(feature_dim, feature_dim, bias=False)
        self.prototypes = nn.Parameter(init_prototypes.clone())
        self.register_buffer("prototype_labels", prototype_labels.clone())
        self.register_buffer("prototype_class_matrix", F.one_hot(prototype_labels, num_classes=num_classes).float())
        self.last_layer = nn.Linear(init_prototypes.shape[0], num_classes)
        self.num_classes = num_classes

        self._initialize_adapter()
        self._initialize_last_layer()

    def _initialize_adapter(self):
        with torch.no_grad():
            eye = torch.eye(self.adapter.weight.shape[0], dtype=self.adapter.weight.dtype)
            self.adapter.weight.copy_(eye)

    def _initialize_last_layer(self):
        with torch.no_grad():
            self.last_layer.weight.fill_(-0.5)
            for proto_idx, class_idx in enumerate(self.prototype_labels.tolist()):
                self.last_layer.weight[class_idx, proto_idx] = 1.0
            self.last_layer.bias.zero_()

    def encode(self, x):
        return F.normalize(self.adapter(x), p=2, dim=1)

    def normalized_prototypes(self):
        return F.normalize(self.prototypes, p=2, dim=1)

    def prototype_distances(self, encoded_features):
        normalized_prototypes = self.normalized_prototypes()
        distances = 2.0 - 2.0 * torch.matmul(encoded_features, normalized_prototypes.T)
        return torch.clamp(distances, min=0.0)

    @staticmethod
    def distance_to_similarity(distances):
        return torch.log((distances + 1.0) / (distances + 1e-4))

    def forward(self, x):
        encoded_features = self.encode(x)
        distances = self.prototype_distances(encoded_features)
        prototype_activations = self.distance_to_similarity(distances)
        logits = self.last_layer(prototype_activations)
        return logits, distances, prototype_activations


def masked_min(distances, mask):
    fill_value = torch.full_like(distances, float("inf"))
    masked_distances = torch.where(mask, distances, fill_value)
    min_distances = masked_distances.min(dim=1).values
    valid_rows = torch.isfinite(min_distances)
    if valid_rows.any():
        return min_distances[valid_rows], valid_rows
    return torch.empty(0, device=distances.device), valid_rows


def compute_loss(model, logits, distances, targets):
    classification_loss = F.binary_cross_entropy_with_logits(logits, targets)

    positive_proto_mask = torch.matmul(targets, model.prototype_class_matrix.T) > 0
    negative_proto_mask = ~positive_proto_mask

    positive_min_distances, positive_valid_rows = masked_min(distances, positive_proto_mask)
    cluster_loss = positive_min_distances.mean() if positive_valid_rows.any() else torch.tensor(0.0, device=distances.device)

    negative_min_distances, negative_valid_rows = masked_min(distances, negative_proto_mask)
    separation_loss = F.relu(SEPARATION_MARGIN - negative_min_distances).mean() if negative_valid_rows.any() else torch.tensor(0.0, device=distances.device)

    off_class_mask = 1.0 - model.prototype_class_matrix.T
    l1_loss = torch.abs(model.last_layer.weight * off_class_mask).mean()

    total_loss = (
        CLASSIFICATION_WEIGHT * classification_loss
        + CLUSTER_WEIGHT * cluster_loss
        + SEPARATION_WEIGHT * separation_loss
        + L1_WEIGHT * l1_loss
    )

    return total_loss, {
        "classification": classification_loss.item(),
        "cluster": cluster_loss.item(),
        "separation": separation_loss.item(),
        "l1": l1_loss.item(),
    }


def make_dataloader(x_tensor, y_tensor, batch_size, shuffle):
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def collect_logits(model, dataloader, device):
    logits_list = []
    with torch.no_grad():
        model.eval()
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            batch_logits, _, _ = model(x_batch)
            logits_list.append(batch_logits.cpu())
    return torch.cat(logits_list, dim=0)


def collect_model_outputs(model, dataloader, device):
    logits_list = []
    distance_list = []
    activation_list = []

    with torch.no_grad():
        model.eval()
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            batch_logits, batch_distances, batch_activations = model(x_batch)
            logits_list.append(batch_logits.cpu())
            distance_list.append(batch_distances.cpu())
            activation_list.append(batch_activations.cpu())

    return torch.cat(logits_list, dim=0), torch.cat(distance_list, dim=0), torch.cat(activation_list, dim=0)


def evaluate_on_loader(model, dataloader, y_true, device, title):
    logits = collect_logits(model, dataloader, device)
    return evaluate_performance(y_true, logits, title=title)


def run_global_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    stats = {"classification": 0.0, "cluster": 0.0, "separation": 0.0, "l1": 0.0}
    steps = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits, distances, _ = model(x_batch)
        loss, loss_stats = compute_loss(model, logits, distances, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for key in stats:
            stats[key] += loss_stats[key]
        steps += 1

    if steps == 0:
        return 0.0, stats

    averaged_stats = {key: value / steps for key, value in stats.items()}
    return total_loss / steps, averaged_stats


def project_prototypes(model, x_train, y_train, device, chunk_size, update_model=True):
    action_text = "Pushing prototypes onto nearest real positive samples"
    if not update_model:
        action_text = "Tracing prototypes to nearest real positive samples"
    print(f"\n[Projection] {action_text} ...")
    model.eval()

    num_prototypes = model.prototypes.shape[0]
    best_distances = torch.full((num_prototypes,), float("inf"), dtype=x_train.dtype, device=device)
    best_features = torch.zeros_like(model.prototypes.data)
    best_indices = torch.full((num_prototypes,), -1, dtype=torch.long, device=device)

    with torch.no_grad():
        for start_idx in range(0, x_train.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, x_train.shape[0])
            x_chunk = x_train[start_idx:end_idx].to(device)
            y_chunk = y_train[start_idx:end_idx].to(device)

            encoded_features = model.encode(x_chunk)
            distances = model.prototype_distances(encoded_features)

            for proto_idx, class_idx in enumerate(model.prototype_labels.tolist()):
                local_mask = y_chunk[:, class_idx] == 1.0
                if not local_mask.any():
                    continue

                candidate_distances = distances[local_mask, proto_idx]
                local_min_idx = torch.argmin(candidate_distances)
                local_best_distance = candidate_distances[local_min_idx]

                if local_best_distance < best_distances[proto_idx]:
                    positive_positions = torch.nonzero(local_mask, as_tuple=False).squeeze(1)
                    selected_chunk_idx = positive_positions[local_min_idx]
                    best_distances[proto_idx] = local_best_distance
                    best_features[proto_idx] = encoded_features[selected_chunk_idx]
                    best_indices[proto_idx] = start_idx + selected_chunk_idx

        valid_mask = best_indices >= 0
        if update_model and valid_mask.any():
            model.prototypes.data[valid_mask] = F.normalize(best_features[valid_mask], p=2, dim=1)

    mapped_count = int(valid_mask.sum().item())
    if update_model:
        print(f"[Projection] Updated {mapped_count}/{num_prototypes} prototypes.")
    else:
        print(f"[Projection] Traced {mapped_count}/{num_prototypes} prototypes.")
    return best_indices.cpu()


def fine_tune_last_layer(model, dataloader, device, num_epochs, lr):
    print(f"[Convex Opt] Fine-tuning last layer for {num_epochs} epochs ...")

    for param in model.adapter.parameters():
        param.requires_grad = False
    model.prototypes.requires_grad_(False)
    for param in model.last_layer.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.last_layer.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, _, _ = model(x_batch)
            classification_loss = F.binary_cross_entropy_with_logits(logits, y_batch)
            off_class_mask = 1.0 - model.prototype_class_matrix.T
            l1_loss = torch.abs(model.last_layer.weight * off_class_mask).mean()
            loss = classification_loss + L1_WEIGHT * l1_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        average_loss = epoch_loss / max(steps, 1)
        print(f"  [Convex Opt] Epoch {epoch}: Loss={average_loss:.4f}")

    for param in model.adapter.parameters():
        param.requires_grad = True
    model.prototypes.requires_grad_(True)


def maybe_save_best(model, val_auc, best_val_auc, best_epoch, epoch, best_model_path):
    if val_auc > best_val_auc:
        torch.save(model.state_dict(), best_model_path)
        print(f"  >>> New best validation AUROC: {best_val_auc:.4f} -> {val_auc:.4f}")
        return val_auc, epoch
    return best_val_auc, best_epoch


def build_prototype_report(model, mapped_train_indices, train_idx):
    rows = []
    labels = model.prototype_labels.detach().cpu().tolist()
    mapped_train_indices = mapped_train_indices.tolist()
    global_train_idx = train_idx.tolist()

    for proto_id, class_idx in enumerate(labels):
        local_idx = mapped_train_indices[proto_id]
        global_idx = global_train_idx[local_idx] if local_idx >= 0 else -1
        rows.append(
            {
                "prototype_id": proto_id,
                "class_idx": class_idx,
                "class_name": TARGET_LABELS[class_idx],
                "train_local_index": local_idx,
                "train_global_index": global_idx,
            }
        )

    return pd.DataFrame(rows)


def analyze_prototype_activation(model, dataloader, device):
    _, _, activations = collect_model_outputs(model, dataloader, device)
    top_proto_indices = torch.argmax(activations, dim=1).numpy()
    counts = np.bincount(top_proto_indices, minlength=activations.shape[1])
    total = counts.sum()

    rows = []
    for proto_id, count in enumerate(counts.tolist()):
        class_idx = int(model.prototype_labels[proto_id].item())
        rows.append(
            {
                "prototype_id": proto_id,
                "class_idx": class_idx,
                "class_name": TARGET_LABELS[class_idx],
                "activation_count": count,
                "activation_rate": (count / total) if total > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values(by=["activation_count", "prototype_id"], ascending=[False, True])


def print_prototype_report(report_df, title):
    print(f"\n=== {title} ===")
    print(f"{'Proto ID':<10} | {'Class':<35} | {'Train Local':<11} | {'Train Global'}")
    print("-" * 85)
    for row in report_df.itertuples(index=False):
        print(
            f"{row.prototype_id:<10} | {row.class_name:<35} | "
            f"{row.train_local_index:<11} | {row.train_global_index}"
        )


def print_activation_report(activation_df, title, top_k=20):
    print(f"\n=== {title} ===")
    print(f"{'Proto ID':<10} | {'Class':<35} | {'Count':<10} | {'Rate'}")
    print("-" * 78)
    for row in activation_df.head(top_k).itertuples(index=False):
        print(
            f"{row.prototype_id:<10} | {row.class_name:<35} | "
            f"{row.activation_count:<10} | {row.activation_rate:.4f}"
        )


def export_prototype_tsne(model, x_reference, y_reference, device, output_path, sample_size):
    print(f"\n[t-SNE] Exporting prototype visualization to {output_path} ...")

    num_reference = min(sample_size, x_reference.shape[0])
    if num_reference <= 10:
        print("[t-SNE] Skipped: not enough reference samples.")
        return

    sample_indices = np.random.choice(x_reference.shape[0], size=num_reference, replace=False)
    x_sample = x_reference[sample_indices].to(device)
    y_sample = y_reference[sample_indices]

    with torch.no_grad():
        encoded_sample = model.encode(x_sample).cpu().numpy()
        encoded_prototypes = model.normalized_prototypes().detach().cpu().numpy()

    combined = np.concatenate([encoded_sample, encoded_prototypes], axis=0)
    perplexity = min(30, max(5, num_reference // 50))
    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    embedding_2d = tsne.fit_transform(combined)

    sample_2d = embedding_2d[:num_reference]
    proto_2d = embedding_2d[num_reference:]
    sample_labels = torch.argmax(y_sample, dim=1).numpy()
    proto_labels = model.prototype_labels.detach().cpu().numpy()

    plt.figure(figsize=(12, 10))
    plt.scatter(
        sample_2d[:, 0],
        sample_2d[:, 1],
        c=sample_labels,
        cmap="tab20",
        alpha=0.18,
        s=8,
        label="HEEDB train samples",
    )
    plt.scatter(
        proto_2d[:, 0],
        proto_2d[:, 1],
        c=proto_labels,
        cmap="tab20",
        edgecolors="black",
        linewidths=0.6,
        s=120,
        marker="X",
        label="Prototypes",
    )

    for proto_id, (x_coord, y_coord) in enumerate(proto_2d):
        plt.text(x_coord, y_coord, f"P{proto_id}", fontsize=7, ha="center", va="bottom")

    plt.title("HEEDB Traditional ProtoPNet Prototypes in Feature Space (t-SNE)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print("[t-SNE] Done.")


def main_ape_traditional_protopnet(args: argparse.Namespace | None = None):
    args = parse_args() if args is None else args
    apply_runtime_config(args)
    set_seed(RANDOM_SEED)
    device = resolve_device(args.device)
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_ROOT_DIR, exist_ok=True)

    print("=== Traditional ProtoPNet Baseline on HEEDB (with MIMIC external test) ===")

    x_train_all, raw_y_train, metadata_train_df = load_and_standardize(DATA_ECG_PATH, DATA_META_PATH, DATA_COL)
    y_train_all = build_ground_truth_matrix(raw_y_train)
    x_train = x_train_all.float()
    y_train = y_train_all.float()
    df_train = attach_export_metadata(
        metadata_train_df.reset_index(drop=True),
        HEEDB_TRAIN_EXPORT_META_PATH,
    )

    x_val_raw, raw_y_val, metadata_val_df = load_and_standardize(VAL_ECG_PATH, VAL_META_PATH, VAL_COL)
    y_val_raw = build_ground_truth_matrix(raw_y_val)
    x_val = x_val_raw.float()
    y_val = y_val_raw.float()

    x_test_raw, raw_y_test, metadata_test_df = load_and_standardize(
        TEST_ROBUST_ECG_PATH,
        TEST_ROBUST_META_PATH,
        TEST_ROBUST_COL,
    )
    y_test_raw = build_ground_truth_matrix(raw_y_test)
    x_test = x_test_raw.float()
    y_test = y_test_raw.float()
    df_test = metadata_test_df.reset_index(drop=True)
    train_idx = np.arange(len(x_train))

    x_mimic_raw, raw_y_mimic, mimic_df = load_and_standardize(TEST_MIMIC_ECG_PATH, TEST_MIMIC_META_PATH, TEST_MIMIC_COL)
    x_mimic = x_mimic_raw.float()
    y_mimic = build_ground_truth_matrix(raw_y_mimic).float()
    mimic_split_bundle = build_mimic_comparable_split(x_mimic, y_mimic, mimic_df)
    x_mimic_split = mimic_split_bundle["x_test"]
    y_mimic_split = mimic_split_bundle["y_test"]

    print(
        f"\nData Split -> Train(Stage1 Dedup): {len(x_train)} | Val(HEEDB Val): {len(x_val)} | "
        f"Test(HEEDB Robust): {len(x_test)} | Test(MIMIC Full): {len(x_mimic)} | "
        f"Test(MIMIC Comparable Split): {len(x_mimic_split)}\n"
    )

    print_classification_prompt_plan()
    prototype_counts = build_prototype_count_plan()
    print_prototype_count_plan(prototype_counts)

    init_prototypes, prototype_labels = initialize_fixed_prototypes(
        x_train,
        y_train,
        prototype_counts,
        INIT_MAX_SAMPLES_PER_CLASS,
    )
    model = TraditionalProtoPNet(init_prototypes, prototype_labels, len(TARGET_LABELS)).to(device)

    train_loader = make_dataloader(x_train, y_train, TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(x_val, y_val, EVAL_BATCH_SIZE, shuffle=False)
    heedb_test_loader = make_dataloader(x_test, y_test, EVAL_BATCH_SIZE, shuffle=False)
    mimic_loader = make_dataloader(x_mimic, y_mimic, EVAL_BATCH_SIZE, shuffle=False)
    mimic_split_loader = make_dataloader(x_mimic_split, y_mimic_split, EVAL_BATCH_SIZE, shuffle=False)

    print("\n" + "=" * 60)
    print(" >> Baseline: Fixed Multi-Prototype Initialization")
    print("=" * 60)
    with torch.no_grad():
        init_logits_internal = collect_logits(model, heedb_test_loader, device)
        baseline_auc_internal, _ = evaluate_performance(
            y_test,
            init_logits_internal,
            title="Initial Fixed-Prototype Baseline on HEEDB Test (Internal)",
        )

        init_logits_mimic = collect_logits(model, mimic_loader, device)
        baseline_auc_mimic_full, _ = evaluate_performance(
            y_mimic,
            init_logits_mimic,
            title="Initial Fixed-Prototype Baseline on MIMIC Test (External Full)",
        )

        init_logits_mimic_split = collect_logits(model, mimic_split_loader, device)
        baseline_auc_mimic_split, _ = evaluate_performance(
            y_mimic_split,
            init_logits_mimic_split,
            title="Initial Fixed-Prototype Baseline on MIMIC Test (External Comparable Split)",
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=GLOBAL_LR)
    best_val_auc = 0.0
    best_epoch = 0

    print("\n" + "=" * 60)
    print(" >> Start Traditional ProtoPNet Training")
    print("=" * 60)
    print(
        f"Training config: epochs={MAX_EPOCHS}, projection_interval={PROJECTION_INTERVAL}, "
        f"last_layer_epochs={LAST_LAYER_FINE_TUNE_EPOCHS}"
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        average_loss, loss_stats = run_global_epoch(model, train_loader, optimizer, device)
        print(
            f"Epoch {epoch}: Loss={average_loss:.4f} | "
            f"cls={loss_stats['classification']:.4f} | "
            f"cluster={loss_stats['cluster']:.4f} | "
            f"sep={loss_stats['separation']:.4f}"
        )

        if epoch % EVAL_INTERVAL == 0:
            val_auc, _ = evaluate_on_loader(
                model,
                val_loader,
                y_val,
                device,
                title=f"Epoch {epoch} Val Monitor",
            )
            best_val_auc, best_epoch = maybe_save_best(
                model,
                val_auc,
                best_val_auc,
                best_epoch,
                epoch,
                BEST_MODEL_PATH,
            )

        if epoch % PROJECTION_INTERVAL == 0:
            project_prototypes(
                model,
                x_train,
                y_train,
                device,
                EVAL_BATCH_SIZE,
                update_model=True,
            )

            val_auc_after_projection, _ = evaluate_on_loader(
                model,
                val_loader,
                y_val,
                device,
                title=f"Epoch {epoch} Val After Projection",
            )
            best_val_auc, best_epoch = maybe_save_best(
                model,
                val_auc_after_projection,
                best_val_auc,
                best_epoch,
                epoch,
                BEST_MODEL_PATH,
            )

            fine_tune_last_layer(
                model,
                train_loader,
                device,
                LAST_LAYER_FINE_TUNE_EPOCHS,
                LAST_LAYER_LR,
            )

            val_auc_after_convex, _ = evaluate_on_loader(
                model,
                val_loader,
                y_val,
                device,
                title=f"Epoch {epoch} Val After Convex Opt",
            )
            best_val_auc, best_epoch = maybe_save_best(
                model,
                val_auc_after_convex,
                best_val_auc,
                best_epoch,
                epoch,
                BEST_MODEL_PATH,
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=GLOBAL_LR)

    print("\n" + "=" * 60)
    print(" >> Final Evaluation (Internal & External)")
    print("=" * 60)

    with torch.no_grad():
        last_logits_internal = collect_logits(model, heedb_test_loader, device)
        last_auc_internal, _ = evaluate_performance(
            y_test,
            last_logits_internal,
            title="Last Traditional ProtoPNet on HEEDB Test (Internal)",
        )

        last_logits_mimic = collect_logits(model, mimic_loader, device)
        last_auc_mimic_full, _ = evaluate_performance(
            y_mimic,
            last_logits_mimic,
            title="Last Traditional ProtoPNet on MIMIC Test (External Full)",
        )

        last_logits_mimic_split = collect_logits(model, mimic_split_loader, device)
        last_auc_mimic_split, _ = evaluate_performance(
            y_mimic_split,
            last_logits_mimic_split,
            title="Last Traditional ProtoPNet on MIMIC Test (External Comparable Split)",
        )
    torch.save(model.state_dict(), LAST_MODEL_PATH)

    best_auc_internal = last_auc_internal
    best_auc_mimic_full = last_auc_mimic_full
    best_auc_mimic_split = last_auc_mimic_split
    if os.path.exists(BEST_MODEL_PATH):
        print(f"\n[Eval] Loading best checkpoint from epoch {best_epoch} ...")
        best_state = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(best_state)

        with torch.no_grad():
            best_logits_internal = collect_logits(model, heedb_test_loader, device)
            best_auc_internal, _ = evaluate_performance(
                y_test,
                best_logits_internal,
                title="Best Traditional ProtoPNet on HEEDB Test (Internal)",
            )

            best_logits_mimic = collect_logits(model, mimic_loader, device)
            best_auc_mimic_full, _ = evaluate_performance(
                y_mimic,
                best_logits_mimic,
                title="Best Traditional ProtoPNet on MIMIC Test (External Full)",
            )

            best_logits_mimic_split = collect_logits(model, mimic_split_loader, device)
            best_auc_mimic_split, _ = evaluate_performance(
                y_mimic_split,
                best_logits_mimic_split,
                title="Best Traditional ProtoPNet on MIMIC Test (External Comparable Split)",
            )
        torch.save(model.state_dict(), BEST_SNAPSHOT_PATH)

    traced_indices = project_prototypes(
        model,
        x_train,
        y_train,
        device,
        EVAL_BATCH_SIZE,
        update_model=False,
    )
    prototype_report_df = build_prototype_report(model, traced_indices, train_idx)
    heedb_activation_df = analyze_prototype_activation(model, heedb_test_loader, device)
    mimic_activation_df = analyze_prototype_activation(model, mimic_loader, device)

    prototype_report_df.to_csv(PROTOTYPE_REPORT_CSV, index=False)
    heedb_activation_df.to_csv(HEEDB_ACTIVATION_CSV, index=False)
    mimic_activation_df.to_csv(MIMIC_ACTIVATION_CSV, index=False)

    print_prototype_report(prototype_report_df, title="Prototype Sample Mapping (HEEDB Train Anchors)")
    print_activation_report(
        heedb_activation_df,
        title="Prototype Activation on HEEDB Test",
        top_k=min(20, len(heedb_activation_df)),
    )
    print_activation_report(
        mimic_activation_df,
        title="Prototype Activation on MIMIC Test",
        top_k=min(20, len(mimic_activation_df)),
    )

    print(f"\n[Report] Saved prototype report to {PROTOTYPE_REPORT_CSV}")
    print(f"[Report] Saved HEEDB activation report to {HEEDB_ACTIVATION_CSV}")
    print(f"[Report] Saved MIMIC activation report to {MIMIC_ACTIVATION_CSV}")

    export_prototype_tsne(model, x_train, y_train, device, PROTOTYPE_TSNE_PNG, TSNE_SAMPLE_SIZE)

    prototype_bank_x, prototype_bank_df = build_prototype_candidate_bank(
        x_train=x_train,
        train_df=df_train,
        prototype_report_df=prototype_report_df,
        activation_report_df=heedb_activation_df,
    )

    if has_exportable_waveform_paths(df_test):
        heedb_json_summary = export_embedding_showcase(
            model=model,
            x_bank=prototype_bank_x,
            bank_df=prototype_bank_df,
            x_query=x_test,
            y_query=y_test,
            query_df=df_test,
            output_base=JSON_EXPORT_DIR_HEEDB,
            device=device,
            query_count=JSON_EXPORT_QUERY_COUNT,
            query_per_class=JSON_EXPORT_QUERY_PER_CLASS,
            top_k=JSON_EXPORT_TOP_K,
            chunk_size=EVAL_BATCH_SIZE,
        )
        print(f"[JSON Export] HEEDB exported {heedb_json_summary['exported_queries']} queries -> {heedb_json_summary['manifest_path']}")
    else:
        heedb_json_summary = {
            "exported_queries": 0,
            "manifest_path": os.path.join(JSON_EXPORT_DIR_HEEDB, "manifest.json"),
            "skipped_reason": "HEEDB robust metadata still has no usable waveform path columns",
        }
        print(
            f"[JSON Export] HEEDB testjson skipped -> {heedb_json_summary['manifest_path']} "
            f"({heedb_json_summary['skipped_reason']})"
        )

    mimic_json_summary = export_embedding_showcase(
        model=model,
        x_bank=prototype_bank_x,
        bank_df=prototype_bank_df,
        x_query=x_mimic,
        y_query=y_mimic,
        query_df=mimic_df,
        output_base=JSON_EXPORT_DIR_MIMIC,
        device=device,
        query_count=JSON_EXPORT_QUERY_COUNT,
        query_per_class=JSON_EXPORT_QUERY_PER_CLASS,
        top_k=JSON_EXPORT_TOP_K,
        chunk_size=EVAL_BATCH_SIZE,
    )
    print(f"[JSON Export] MIMIC exported {mimic_json_summary['exported_queries']} queries -> {mimic_json_summary['manifest_path']}")

    print("\n=== Summary ===")
    print("-" * 96)
    print(f"{'Dataset':<25} | {'Baseline':<10} | {'Last':<10} | {'Best':<10}")
    print("-" * 96)
    print(
        f"{'HEEDB (Internal)':<25} | {baseline_auc_internal:.4f}     | "
        f"{last_auc_internal:.4f}     | {best_auc_internal:.4f}"
    )
    print(
        f"{'MIMIC (External Full)':<25} | {baseline_auc_mimic_full:.4f}     | "
        f"{last_auc_mimic_full:.4f}     | {best_auc_mimic_full:.4f}"
    )
    print(
        f"{'MIMIC (External Split)':<25} | {baseline_auc_mimic_split:.4f}     | "
        f"{last_auc_mimic_split:.4f}     | {best_auc_mimic_split:.4f}"
    )
    print("-" * 96)
    print(f"Best validation epoch: {best_epoch}")
    print(f"Total prototypes: {model.prototypes.shape[0]}")
    print(f"Prototype report:      {PROTOTYPE_REPORT_CSV}")
    print(f"HEEDB activations:     {HEEDB_ACTIVATION_CSV}")
    print(f"MIMIC activations:     {MIMIC_ACTIVATION_CSV}")
    print(f"Prototype t-SNE:       {PROTOTYPE_TSNE_PNG}")
    print(f"HEEDB JSON manifest:   {heedb_json_summary['manifest_path']}")
    print(f"MIMIC JSON manifest:   {mimic_json_summary['manifest_path']}")


if __name__ == "__main__":
    main_ape_traditional_protopnet()
