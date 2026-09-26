#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CODE-15 prototype base implementation.

Purpose
-------
Provide CODE-15 record-level data, label construction, prototype architecture,
and training helpers.

Method
------
Read binary/multilabel tensors, create seeded splits, initialize class-
positive prototypes, project to real cases, and export activation reports.

Inputs
------
CODE-15 ECG vectors, label tensor, metadata/exam IDs, and external output
paths.

Outputs
-------
External checkpoint files, prototype reports, and score/class metrics.

Paper parameters
----------------
The formal wrapper supplies the paper's six-label task and budget; alternate
base defaults are retained for helper reuse.

Shapes and meaning
------------------
ECG inputs are [N,D], targets [N,6] in the paper multilabel task, prototypes
[P,D], and output logits [N,6].
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

from training_json_export_bridge import build_prototype_candidate_bank, export_embedding_showcase

warnings.filterwarnings("ignore")


EXPERIMENT_NAME = "code15_traditional"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("ECG_OUTPUT_ROOT", "__OUTPUT_DIR_REQUIRED__")) / "protopnet" / EXPERIMENT_NAME

CODE15_ECG_PATH = "__ECG_DATA_ROOT_REQUIRED__/ecofounder2_ex/code15/code15_ecgclip_embeddings.pt"
CODE15_LABELS_PATH = "__ECG_DATA_ROOT_REQUIRED__/ecofounder2_ex/code15/code15_labels.pt"
CODE15_META_PATH = "__ECG_DATA_ROOT_REQUIRED__/ecofounder2_ex/code15/metadata_code15_ecgclip.csv"

TARGET_LABELS = [
    "Chagas disease",
    "1st degree atrioventricular block",
    "Right bundle branch block",
    "Left bundle branch block",
    "Sinus bradycardia",
    "Sinus tachycardia",
    "Atrial fibrillation",
]

DEFAULT_PROTOTYPES_PER_CLASS = 3
PROTOTYPE_COUNT_OVERRIDES = {
    "1st degree atrioventricular block": 10,
    "Atrial fibrillation": 6,
    "Right bundle branch block": 5,
    "Left bundle branch block": 5,
    "Chagas disease": 5,
}

TRAIN_BATCH_SIZE = 4096
EVAL_BATCH_SIZE = 8192
INIT_MAX_SAMPLES_PER_CLASS = 30000
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
TSNE_SAMPLE_SIZE = 3000

OUTPUT_ROOT_DIR = str(DEFAULT_OUTPUT_ROOT)
AUTO_ROOT_DIR = OUTPUT_ROOT_DIR
CHECKPOINT_ROOT_DIR = os.path.join(OUTPUT_ROOT_DIR, "checkpoint")

BEST_MODEL_PATH = os.path.join(CHECKPOINT_ROOT_DIR, "code15_traditional_protopnet_best.pt")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_ROOT_DIR, "code15_traditional_protopnet_last.pt")
BEST_SNAPSHOT_PATH = os.path.join(CHECKPOINT_ROOT_DIR, "code15_traditional_protopnet_best_snapshot.pt")
PROTOTYPE_REPORT_CSV = os.path.join(OUTPUT_ROOT_DIR, "code15_traditional_protopnet_prototype_report.csv")
ACTIVATION_REPORT_CSV = os.path.join(OUTPUT_ROOT_DIR, "code15_traditional_protopnet_activation_report.csv")
PROTOTYPE_TSNE_PNG = os.path.join(OUTPUT_ROOT_DIR, "code15_traditional_protopnet_tsne.png")
JSON_EXPORT_DIR = os.path.join(OUTPUT_ROOT_DIR, "code15_traditional_protopnet_json")
JSON_EXPORT_QUERY_COUNT = 12
JSON_EXPORT_QUERY_PER_CLASS = 2
JSON_EXPORT_TOP_K = 8


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traditional ProtoPNet-style CODE-15 training.")
    parser.add_argument("--code15-ecg-path", default=CODE15_ECG_PATH)
    parser.add_argument("--code15-labels-path", default=CODE15_LABELS_PATH)
    parser.add_argument("--code15-meta-path", default=CODE15_META_PATH)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def apply_runtime_config(args: argparse.Namespace) -> None:
    global CODE15_ECG_PATH, CODE15_LABELS_PATH, CODE15_META_PATH
    global OUTPUT_ROOT_DIR, AUTO_ROOT_DIR, CHECKPOINT_ROOT_DIR
    global BEST_MODEL_PATH, LAST_MODEL_PATH, BEST_SNAPSHOT_PATH
    global PROTOTYPE_REPORT_CSV, ACTIVATION_REPORT_CSV, PROTOTYPE_TSNE_PNG, JSON_EXPORT_DIR
    global RANDOM_SEED

    CODE15_ECG_PATH = args.code15_ecg_path
    CODE15_LABELS_PATH = args.code15_labels_path
    CODE15_META_PATH = args.code15_meta_path
    RANDOM_SEED = int(args.seed)

    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else output_root / "checkpoint"
    OUTPUT_ROOT_DIR = str(output_root)
    AUTO_ROOT_DIR = OUTPUT_ROOT_DIR
    CHECKPOINT_ROOT_DIR = str(checkpoint_root)
    BEST_MODEL_PATH = str(checkpoint_root / "code15_traditional_protopnet_best.pt")
    LAST_MODEL_PATH = str(checkpoint_root / "code15_traditional_protopnet_last.pt")
    BEST_SNAPSHOT_PATH = str(checkpoint_root / "code15_traditional_protopnet_best_snapshot.pt")
    PROTOTYPE_REPORT_CSV = str(output_root / "code15_traditional_protopnet_prototype_report.csv")
    ACTIVATION_REPORT_CSV = str(output_root / "code15_traditional_protopnet_activation_report.csv")
    PROTOTYPE_TSNE_PNG = str(output_root / "code15_traditional_protopnet_tsne.png")
    JSON_EXPORT_DIR = str(output_root / "code15_traditional_protopnet_json")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset():
    print(f"--> Loading CODE-15 embeddings from {CODE15_ECG_PATH} ...")
    ecg_emb = torch.load(CODE15_ECG_PATH, map_location="cpu")
    ecg_emb = F.normalize(ecg_emb, p=2, dim=1)
    ecg_emb = torch.nan_to_num(ecg_emb, nan=0.0)

    print(f"--> Loading predefined labels from {CODE15_LABELS_PATH} ...")
    labels = torch.load(CODE15_LABELS_PATH, map_location="cpu").float()

    metadata_df = pd.read_csv(CODE15_META_PATH)

    if len(labels) != len(ecg_emb) or len(metadata_df) != len(ecg_emb):
        print(
            f"Warning: Metadata len {len(metadata_df)} / Labels len {len(labels)} "
            f"!= Emb len {len(ecg_emb)}"
        )
        min_len = min(len(labels), len(ecg_emb), len(metadata_df))
        ecg_emb = ecg_emb[:min_len]
        labels = labels[:min_len]
        metadata_df = metadata_df.iloc[:min_len]

    return ecg_emb, labels, metadata_df.reset_index(drop=True)


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
    return [
        PROTOTYPE_COUNT_OVERRIDES.get(label_name, DEFAULT_PROTOTYPES_PER_CLASS)
        for label_name in TARGET_LABELS
    ]


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
        self.register_buffer(
            "prototype_class_matrix",
            F.one_hot(prototype_labels, num_classes=num_classes).float(),
        )
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

    return (
        torch.cat(logits_list, dim=0),
        torch.cat(distance_list, dim=0),
        torch.cat(activation_list, dim=0),
    )


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
        cmap="tab10",
        alpha=0.18,
        s=8,
        label="CODE-15 train samples",
    )
    plt.scatter(
        proto_2d[:, 0],
        proto_2d[:, 1],
        c=proto_labels,
        cmap="tab10",
        edgecolors="black",
        linewidths=0.6,
        s=120,
        marker="X",
        label="Prototypes",
    )

    for proto_id, (x_coord, y_coord) in enumerate(proto_2d):
        plt.text(x_coord, y_coord, f"P{proto_id}", fontsize=7, ha="center", va="bottom")

    plt.title("CODE-15 Traditional ProtoPNet Prototypes in Feature Space (t-SNE)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print("[t-SNE] Done.")


def main_code15_traditional_protopnet(args: argparse.Namespace | None = None):
    args = parse_args() if args is None else args
    apply_runtime_config(args)
    set_seed(RANDOM_SEED)
    device = resolve_device(args.device)
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_ROOT_DIR, exist_ok=True)

    print("=== Traditional ProtoPNet Baseline on CODE-15 ===")

    x_all, y_all, metadata_df = load_dataset()
    indices = np.arange(len(x_all))
    train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=RANDOM_SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=RANDOM_SEED)

    x_train = x_all[train_idx].float()
    y_train = y_all[train_idx].float()
    x_val = x_all[val_idx].float()
    y_val = y_all[val_idx].float()
    x_test = x_all[test_idx].float()
    y_test = y_all[test_idx].float()
    df_train = metadata_df.iloc[train_idx].reset_index(drop=True)
    df_test = metadata_df.iloc[test_idx].reset_index(drop=True)

    print(f"\nSplit Summary:\n Train: {len(x_train)}\n Val:   {len(x_val)}\n Test:  {len(x_test)}\n")

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
    test_loader = make_dataloader(x_test, y_test, EVAL_BATCH_SIZE, shuffle=False)

    print("\n" + "=" * 60)
    print(" >> Baseline: Fixed Multi-Prototype Initialization")
    print("=" * 60)
    with torch.no_grad():
        init_logits = collect_logits(model, test_loader, device)
        init_auc, _ = evaluate_performance(y_test, init_logits, title="Initial Fixed-Prototype Baseline on CODE-15 Test")

    optimizer = optim.Adam(model.parameters(), lr=GLOBAL_LR)
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

            optimizer = optim.Adam(model.parameters(), lr=GLOBAL_LR)

    print("\n" + "=" * 60)
    print(" >> Final Evaluation")
    print("=" * 60)

    with torch.no_grad():
        last_logits = collect_logits(model, test_loader, device)
        last_auc, _ = evaluate_performance(y_test, last_logits, title="Last Traditional ProtoPNet on CODE-15 Test")
    torch.save(model.state_dict(), LAST_MODEL_PATH)

    best_auc = last_auc
    if os.path.exists(BEST_MODEL_PATH):
        print(f"\n[Eval] Loading best checkpoint from epoch {best_epoch} ...")
        best_state = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(best_state)
        with torch.no_grad():
            best_logits = collect_logits(model, test_loader, device)
            best_auc, _ = evaluate_performance(y_test, best_logits, title="Best Traditional ProtoPNet on CODE-15 Test")
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
    activation_df = analyze_prototype_activation(model, test_loader, device)

    prototype_report_df.to_csv(PROTOTYPE_REPORT_CSV, index=False)
    activation_df.to_csv(ACTIVATION_REPORT_CSV, index=False)

    print_prototype_report(prototype_report_df, title="Prototype Sample Mapping (CODE-15 Train Anchors)")
    print_activation_report(activation_df, title="Prototype Activation on CODE-15 Test", top_k=min(20, len(activation_df)))

    print(f"\n[Report] Saved prototype report to {PROTOTYPE_REPORT_CSV}")
    print(f"[Report] Saved activation report to {ACTIVATION_REPORT_CSV}")

    export_prototype_tsne(model, x_train, y_train, device, PROTOTYPE_TSNE_PNG, TSNE_SAMPLE_SIZE)

    prototype_bank_x, prototype_bank_df = build_prototype_candidate_bank(
        x_train=x_train,
        train_df=df_train,
        prototype_report_df=prototype_report_df,
        activation_report_df=activation_df,
    )

    json_export_summary = export_embedding_showcase(
        model=model,
        x_bank=prototype_bank_x,
        bank_df=prototype_bank_df,
        x_query=x_test,
        y_query=y_test,
        query_df=df_test,
        output_base=JSON_EXPORT_DIR,
        device=device,
        query_count=JSON_EXPORT_QUERY_COUNT,
        query_per_class=JSON_EXPORT_QUERY_PER_CLASS,
        top_k=JSON_EXPORT_TOP_K,
        chunk_size=EVAL_BATCH_SIZE,
    )
    print(f"[JSON Export] Exported {json_export_summary['exported_queries']} queries -> {json_export_summary['manifest_path']}")

    print("\n=== Summary ===")
    print("-" * 72)
    print(f"{'Model':<38} | {'AUROC':<10} | {'Delta vs Init':<14}")
    print("-" * 72)
    print(f"{'Initial Fixed-Prototype Baseline':<38} | {init_auc:.4f}     | {'-':<14}")
    print(f"{'Last Traditional ProtoPNet':<38} | {last_auc:.4f}     | {(last_auc - init_auc) * 100:+.2f}%")
    print(f"{'Best Traditional ProtoPNet':<38} | {best_auc:.4f}     | {(best_auc - init_auc) * 100:+.2f}%")
    print("-" * 72)
    print(f"Best validation epoch: {best_epoch}")
    print(f"Total prototypes: {model.prototypes.shape[0]}")
    print(f"Prototype report: {PROTOTYPE_REPORT_CSV}")
    print(f"Activation report:{ACTIVATION_REPORT_CSV}")
    print(f"Prototype t-SNE:  {PROTOTYPE_TSNE_PNG}")
    print(f"JSON manifest:    {json_export_summary['manifest_path']}")


if __name__ == "__main__":
    main_code15_traditional_protopnet()
