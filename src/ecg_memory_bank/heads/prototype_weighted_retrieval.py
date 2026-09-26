#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared prototype model components.

Purpose
-------
Implement the cosine-similarity prototype bank and three score-aggregation
variants used by CODE-15 and related tasks.

Method
------
Allocate positive-class prototypes, transform embeddings, compute signed
similarities, project anchors, and train with
classification/cluster/separation losses.

Inputs
------
Frozen ECG vectors, binary target matrix, class prototype plan, and optimizer
settings from a task wrapper.

Outputs
-------
Model objects, class scores, projection indices, and training/evaluation
summaries.

Paper parameters
----------------
Prototype count and regularization settings are set by the formal task; only
projected real cases can serve as audit anchors.

Shapes and meaning
------------------
Input is [N,D], prototypes [P,D], similarities [N,P], and class scores [N,C].
Signed contributions can support or suppress a logit.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = BASE_DIR.parent
LOCAL_COMMON_DIR = EXPERIMENT_ROOT / "common"
LOCAL_PROTOPNET_DIR = EXPERIMENT_ROOT / "protopnet"

for candidate in (LOCAL_PROTOPNET_DIR, LOCAL_COMMON_DIR):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.append(str(candidate))

from ecg_memory_bank.heads.label_mapping_utils import normalize_label_text  # noqa: E402


def load_shared_base():
    # Optional local compatibility hook. The parallel branch is designed to run
    # without the old shared baseline, so all required defaults have fallbacks.
    module_path = LOCAL_PROTOPNET_DIR / "prototype_heedb.py"
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("prototype_new_shared_traditional_base", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SHARED_BASE = load_shared_base()

DEFAULT_PROTOTYPES_PER_CLASS = int(getattr(SHARED_BASE, "DEFAULT_PROTOTYPES_PER_CLASS", 3))
PROTOTYPE_COUNT_OVERRIDES = dict(getattr(SHARED_BASE, "PROTOTYPE_COUNT_OVERRIDES", {}))
TRAIN_BATCH_SIZE = int(getattr(SHARED_BASE, "TRAIN_BATCH_SIZE", 8192))
EVAL_BATCH_SIZE = int(getattr(SHARED_BASE, "EVAL_BATCH_SIZE", 16384))
INIT_MAX_SAMPLES_PER_CLASS = int(getattr(SHARED_BASE, "INIT_MAX_SAMPLES_PER_CLASS", 50000))
MAX_EPOCHS = int(getattr(SHARED_BASE, "MAX_EPOCHS", 20))
EVAL_INTERVAL = int(getattr(SHARED_BASE, "EVAL_INTERVAL", 2))
PROJECTION_INTERVAL = int(getattr(SHARED_BASE, "PROJECTION_INTERVAL", 5))
LAST_LAYER_FINE_TUNE_EPOCHS = int(getattr(SHARED_BASE, "LAST_LAYER_FINE_TUNE_EPOCHS", 3))
GLOBAL_LR = float(getattr(SHARED_BASE, "GLOBAL_LR", 1e-3))
LAST_LAYER_LR = float(getattr(SHARED_BASE, "LAST_LAYER_LR", 2e-3))
CLASSIFICATION_WEIGHT = float(getattr(SHARED_BASE, "CLASSIFICATION_WEIGHT", 1.0))
CLUSTER_WEIGHT = float(getattr(SHARED_BASE, "CLUSTER_WEIGHT", 0.8))
SEPARATION_WEIGHT = float(getattr(SHARED_BASE, "SEPARATION_WEIGHT", 0.08))
SEPARATION_MARGIN = float(getattr(SHARED_BASE, "SEPARATION_MARGIN", 1.0))
RANDOM_SEED = int(getattr(SHARED_BASE, "RANDOM_SEED", 42))
TSNE_SAMPLE_SIZE = int(getattr(SHARED_BASE, "TSNE_SAMPLE_SIZE", 4000))

WEIGHT_L1_REGULARIZER_WEIGHT = 1e-3
WEIGHT_L2_REGULARIZER_WEIGHT = 1e-4
INITIAL_PROTOTYPE_WEIGHT = 1.0
NO_CLASS_LOGIT = -1e4
PROJECTION_TIE_ATOL = 1e-8
PROJECTION_TIE_RTOL = 1e-5
POSTHOC_LINEAR_HEAD_LR = 2e-3
POSTHOC_LINEAR_HEAD_EPOCHS = 20
POSTHOC_LINEAR_HEAD_L1_WEIGHT = 1e-4
CLASS_TOP_K_VOTE = 5
PRUNE_WEIGHT_ABS_THRESHOLD = 0.05
TRAIN_EXPLORATION_NOISE_STD = 0.1
ENABLE_TRAIN_EXPLORATION = False
PRUNE_MIN_KEEP_PER_CLASS = 1

TARGET_LABELS = list(getattr(SHARED_BASE, "TARGET_LABELS", [])) if SHARED_BASE is not None else []
TARGET_KEYWORDS = list(getattr(SHARED_BASE, "TARGET_KEYWORDS", [])) if SHARED_BASE is not None else []
TARGET_PROMPTS = list(getattr(SHARED_BASE, "TARGET_PROMPTS", [])) if SHARED_BASE is not None else []
TARGET_ALIAS_SETS = list(getattr(SHARED_BASE, "TARGET_ALIAS_SETS", [])) if SHARED_BASE is not None else []
NORMAL_LABEL = str(getattr(SHARED_BASE, "NORMAL_LABEL", "Normal ECG")) if SHARED_BASE is not None else "Normal ECG"
NORMAL_CLASS_IDX = int(getattr(SHARED_BASE, "NORMAL_CLASS_IDX", -1)) if SHARED_BASE is not None else -1


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def build_prototype_count_plan():
    return [PROTOTYPE_COUNT_OVERRIDES.get(label_name, DEFAULT_PROTOTYPES_PER_CLASS) for label_name in TARGET_LABELS]


def initialize_fixed_prototypes(x_train, y_train, counts, max_samples_per_class):
    print("\n--> Initializing fixed class-specific prototypes for signed-weight retrieval ...")
    prototype_vectors = []
    prototype_label_rows = []
    feature_dim = x_train.shape[1]
    num_classes = int(y_train.shape[1])

    for class_idx, num_prototypes in enumerate(counts):
        positive_mask = y_train[:, class_idx] == 1.0
        positive_samples = x_train[positive_mask]
        label_name = TARGET_LABELS[class_idx]

        if positive_samples.shape[0] == 0:
            print(f"  [Warning] {label_name}: no positive samples. Using random unit vectors.")
            class_prototypes = torch.randn(num_prototypes, feature_dim, dtype=x_train.dtype)
            class_prototypes = F.normalize(class_prototypes, p=2, dim=1)
        else:
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

        class_label_rows = torch.zeros((num_prototypes, num_classes), dtype=torch.float32)
        class_label_rows[:, class_idx] = 1.0
        prototype_vectors.append(class_prototypes)
        prototype_label_rows.append(class_label_rows)
        print(f"  {label_name:<35} | positives={positive_samples.shape[0]:<8} | prototypes={num_prototypes}")

    return torch.cat(prototype_vectors, dim=0), torch.cat(prototype_label_rows, dim=0)


class TraditionalProtoPNetComparator(nn.Module):
    def __init__(self, init_prototypes: torch.Tensor, prototype_labels: torch.Tensor, num_classes: int):
        super().__init__()
        feature_dim = init_prototypes.shape[1]

        self.adapter = nn.Linear(feature_dim, feature_dim, bias=False)
        self.prototypes = nn.Parameter(init_prototypes.clone())
        self.register_buffer("prototype_labels", prototype_labels.clone().long())
        self.register_buffer("prototype_class_matrix", F.one_hot(self.prototype_labels, num_classes=num_classes).float())
        self.last_layer = nn.Linear(init_prototypes.shape[0], num_classes)
        self.num_classes = int(num_classes)

        self._initialize_adapter()
        self._initialize_last_layer()

    def _initialize_adapter(self):
        with torch.no_grad():
            eye = torch.eye(self.adapter.weight.shape[0], dtype=self.adapter.weight.dtype, device=self.adapter.weight.device)
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

    def prototype_onclass_weights(self) -> torch.Tensor:
        proto_indices = torch.arange(self.prototype_labels.shape[0], device=self.prototype_labels.device)
        return self.last_layer.weight[self.prototype_labels, proto_indices]

    def count_weight_signs(self) -> dict[str, int]:
        with torch.no_grad():
            weights = self.prototype_onclass_weights().detach().cpu().view(-1)
            return {
                "active_total": int(len(weights)),
                "negative_count": int((weights < -PRUNE_WEIGHT_ABS_THRESHOLD).sum().item()),
                "positive_count": int((weights > PRUNE_WEIGHT_ABS_THRESHOLD).sum().item()),
                "near_zero_count": int((torch.abs(weights) <= PRUNE_WEIGHT_ABS_THRESHOLD).sum().item()),
            }


class ConfidenceWeightedProtoPNet(nn.Module):
    def __init__(
        self,
        init_prototypes: torch.Tensor,
        prototype_label_matrix: torch.Tensor,
        num_classes: int,
        initial_weight: float = INITIAL_PROTOTYPE_WEIGHT,
    ):
        super().__init__()
        feature_dim = init_prototypes.shape[1]
        num_prototypes = init_prototypes.shape[0]

        if prototype_label_matrix.shape != (num_prototypes, num_classes):
            raise ValueError(
                "prototype_label_matrix shape mismatch: "
                f"expected {(num_prototypes, num_classes)}, got {tuple(prototype_label_matrix.shape)}"
            )

        self.adapter = nn.Linear(feature_dim, feature_dim, bias=False)
        self.prototypes = nn.Parameter(init_prototypes.clone())
        self.register_buffer("prototype_label_matrix", prototype_label_matrix.clone().float())
        self.register_buffer("prototype_primary_label", torch.argmax(self.prototype_label_matrix, dim=1).long())
        self.register_buffer("prototype_active_mask", torch.ones(num_prototypes, dtype=torch.bool))
        self.register_buffer("prototype_epoch_activation_counts", torch.zeros(num_prototypes, dtype=torch.long))
        self.num_classes = int(num_classes)
        self.prototype_coefficients = nn.Parameter(torch.full((num_prototypes, 1), float(initial_weight), dtype=init_prototypes.dtype))
        self.training_exploration_noise_std = float(TRAIN_EXPLORATION_NOISE_STD)
        self.enable_training_exploration = bool(ENABLE_TRAIN_EXPLORATION)

        self._initialize_adapter()
        self._refresh_primary_labels()

    def _initialize_adapter(self):
        with torch.no_grad():
            eye = torch.eye(self.adapter.weight.shape[0], dtype=self.adapter.weight.dtype, device=self.adapter.weight.device)
            self.adapter.weight.copy_(eye)

    def _refresh_primary_labels(self):
        with torch.no_grad():
            label_counts = self.prototype_label_matrix.sum(dim=1)
            primary = torch.argmax(self.prototype_label_matrix, dim=1)
            primary = torch.where(label_counts > 0, primary, torch.zeros_like(primary))
            self.prototype_primary_label.copy_(primary.long())

    def update_prototype_label_matrix(self, new_matrix: torch.Tensor) -> None:
        if new_matrix.shape != self.prototype_label_matrix.shape:
            raise ValueError(
                "new prototype label matrix shape mismatch: "
                f"expected {tuple(self.prototype_label_matrix.shape)}, got {tuple(new_matrix.shape)}"
            )
        with torch.no_grad():
            self.prototype_label_matrix.copy_(new_matrix.to(device=self.prototype_label_matrix.device, dtype=self.prototype_label_matrix.dtype))
            self._refresh_primary_labels()

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

    def prototype_weights(self):
        return self.prototype_coefficients

    def set_training_exploration(self, enabled: bool, noise_std: float | None = None):
        self.enable_training_exploration = bool(enabled)
        if noise_std is not None:
            self.training_exploration_noise_std = float(noise_std)

    def reset_epoch_activation_counts(self):
        with torch.no_grad():
            self.prototype_epoch_activation_counts.zero_()

    def get_epoch_activation_counts(self) -> torch.Tensor:
        return self.prototype_epoch_activation_counts.detach().clone()

    def count_weight_signs(self) -> dict[str, int]:
        with torch.no_grad():
            weights = self.prototype_weights().detach().view(-1)
            active_mask = self.prototype_active_mask.detach()
            active_weights = weights[active_mask]
            return {
                "active_total": int(active_mask.sum().item()),
                "negative_count": int((active_weights < -PRUNE_WEIGHT_ABS_THRESHOLD).sum().item()),
                "positive_count": int((active_weights > PRUNE_WEIGHT_ABS_THRESHOLD).sum().item()),
                "near_zero_count": int((torch.abs(active_weights) <= PRUNE_WEIGHT_ABS_THRESHOLD).sum().item()),
            }

    def prune_inactive_prototypes(self, min_activation_count: int = 1, min_keep_per_class: int = PRUNE_MIN_KEEP_PER_CLASS) -> dict[str, object]:
        with torch.no_grad():
            counts = self.prototype_epoch_activation_counts.detach().clone()
            active_mask = self.prototype_active_mask.detach().clone()
            candidate_mask = counts < int(min_activation_count)
            new_active_mask = active_mask.clone()
            new_active_mask[candidate_mask] = False

            for class_idx in range(self.num_classes):
                class_mask = self.prototype_label_matrix[:, class_idx] > 0
                currently_active = torch.nonzero(new_active_mask & class_mask, as_tuple=False).squeeze(1)
                if currently_active.numel() >= int(min_keep_per_class):
                    continue
                fallback_candidates = torch.nonzero(active_mask & class_mask, as_tuple=False).squeeze(1)
                if fallback_candidates.numel() == 0:
                    continue
                fallback_weights = torch.abs(self.prototype_weights().detach().view(-1)[fallback_candidates])
                order = torch.argsort(fallback_weights, descending=True)
                needed = int(min_keep_per_class) - int(currently_active.numel())
                for idx in order.tolist():
                    proto_idx = int(fallback_candidates[int(idx)].item())
                    new_active_mask[proto_idx] = True
                    needed -= 1
                    if needed <= 0:
                        break

            pruned_mask = active_mask & ~new_active_mask
            self.prototype_active_mask.copy_(new_active_mask)
            if bool(pruned_mask.any()):
                self.prototype_coefficients.data[pruned_mask] = 0.0

            payload = {
                "min_activation_count": int(min_activation_count),
                "pruned_count": int(pruned_mask.sum().item()),
                "remaining_active_count": int(new_active_mask.sum().item()),
                "pruned_indices": torch.nonzero(pruned_mask, as_tuple=False).squeeze(1).cpu().tolist(),
            }
            self.prototype_epoch_activation_counts.zero_()
            return payload

    def class_vote_logits(self, signed_activations: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (signed_activations.shape[0], self.num_classes),
            NO_CLASS_LOGIT,
            dtype=signed_activations.dtype,
            device=signed_activations.device,
        )
        class_membership = (self.prototype_label_matrix.T > 0) & self.prototype_active_mask.unsqueeze(0)
        for class_idx in range(self.num_classes):
            member_mask = class_membership[class_idx]
            if not bool(member_mask.any()):
                continue
            class_signed = signed_activations[:, member_mask]
            k = min(CLASS_TOP_K_VOTE, class_signed.shape[1])
            if k <= 0:
                continue
            selection_scores = torch.abs(class_signed)
            if self.training and self.enable_training_exploration and self.training_exploration_noise_std > 0:
                selection_scores = selection_scores + torch.randn_like(selection_scores) * self.training_exploration_noise_std
            top_indices = torch.topk(selection_scores, k=k, dim=1).indices
            top_values = torch.gather(class_signed, dim=1, index=top_indices)
            logits[:, class_idx] = top_values.sum(dim=1)
            if self.training:
                with torch.no_grad():
                    member_indices = torch.nonzero(member_mask, as_tuple=False).squeeze(1)
                    selected_proto_indices = member_indices[top_indices.reshape(-1)].cpu()
                    if selected_proto_indices.numel() > 0:
                        bincount = torch.bincount(selected_proto_indices, minlength=self.prototype_epoch_activation_counts.numel())
                        self.prototype_epoch_activation_counts += bincount.to(self.prototype_epoch_activation_counts.device)
        return logits

    def forward(self, x):
        encoded_features = self.encode(x)
        distances = self.prototype_distances(encoded_features)
        raw_activations = self.distance_to_similarity(distances)
        signed_activations = raw_activations * self.prototype_weights().T
        logits = self.class_vote_logits(signed_activations)
        return logits, distances, signed_activations


class ConfidenceWeightedWithLinearHead(nn.Module):
    def __init__(self, base_model: ConfidenceWeightedProtoPNet):
        super().__init__()
        self.base_model = base_model
        self.linear_head = nn.Linear(base_model.prototypes.shape[0], base_model.num_classes)
        self.register_buffer("prototype_class_matrix", base_model.prototype_label_matrix.detach().clone().T)

        with torch.no_grad():
            self.linear_head.weight.fill_(-0.5)
            self.linear_head.bias.zero_()
            label_matrix = base_model.prototype_label_matrix.detach().cpu()
            for proto_idx in range(label_matrix.shape[0]):
                active_classes = torch.nonzero(label_matrix[proto_idx] > 0, as_tuple=False).squeeze(1).tolist()
                if not active_classes:
                    continue
                share = 1.0 / float(len(active_classes))
                for class_idx in active_classes:
                    self.linear_head.weight[int(class_idx), proto_idx] = share

    def forward(self, x):
        _base_logits, distances, weighted_activations = self.base_model(x)
        logits = self.linear_head(weighted_activations)
        return logits, distances, weighted_activations


def masked_min(distances, mask):
    fill_value = torch.full_like(distances, float("inf"))
    masked_distances = torch.where(mask, distances, fill_value)
    min_distances = masked_distances.min(dim=1).values
    valid_rows = torch.isfinite(min_distances)
    if valid_rows.any():
        return min_distances[valid_rows], valid_rows
    return torch.empty(0, device=distances.device), valid_rows


def compute_weight_regularizer(model: ConfidenceWeightedProtoPNet) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = model.prototype_weights()
    l1_term = torch.abs(weights).mean()
    l2_term = torch.square(weights).mean()
    total = WEIGHT_L1_REGULARIZER_WEIGHT * l1_term + WEIGHT_L2_REGULARIZER_WEIGHT * l2_term
    return total, l1_term, l2_term


def compute_loss(model, logits, distances, targets):
    classification_loss = F.binary_cross_entropy_with_logits(logits, targets)

    positive_proto_mask = torch.matmul(targets, model.prototype_label_matrix.T) > 0
    negative_proto_mask = torch.matmul(1.0 - targets, model.prototype_label_matrix.T) > 0

    positive_min_distances, positive_valid_rows = masked_min(distances, positive_proto_mask)
    cluster_loss = positive_min_distances.mean() if positive_valid_rows.any() else torch.tensor(0.0, device=distances.device)

    negative_min_distances, negative_valid_rows = masked_min(distances, negative_proto_mask)
    separation_loss = (
        F.relu(SEPARATION_MARGIN - negative_min_distances).mean()
        if negative_valid_rows.any()
        else torch.tensor(0.0, device=distances.device)
    )

    weight_regularizer, weight_l1, weight_l2 = compute_weight_regularizer(model)
    total_loss = (
        CLASSIFICATION_WEIGHT * classification_loss
        + CLUSTER_WEIGHT * cluster_loss
        + SEPARATION_WEIGHT * separation_loss
        + weight_regularizer
    )

    return total_loss, {
        "classification": float(classification_loss.item()),
        "cluster": float(cluster_loss.item()),
        "separation": float(separation_loss.item()),
        "weight_l1": float(weight_l1.item()),
        "weight_l2": float(weight_l2.item()),
    }


def compute_traditional_loss(model, logits, distances, targets):
    classification_loss = F.binary_cross_entropy_with_logits(logits, targets)

    positive_proto_mask = torch.matmul(targets, model.prototype_class_matrix.T) > 0
    negative_proto_mask = ~positive_proto_mask

    positive_min_distances, positive_valid_rows = masked_min(distances, positive_proto_mask)
    cluster_loss = positive_min_distances.mean() if positive_valid_rows.any() else torch.tensor(0.0, device=distances.device)

    negative_min_distances, negative_valid_rows = masked_min(distances, negative_proto_mask)
    separation_loss = (
        F.relu(SEPARATION_MARGIN - negative_min_distances).mean()
        if negative_valid_rows.any()
        else torch.tensor(0.0, device=distances.device)
    )

    off_class_mask = 1.0 - model.prototype_class_matrix.T
    l1_loss = torch.abs(model.last_layer.weight * off_class_mask).mean()

    total_loss = (
        CLASSIFICATION_WEIGHT * classification_loss
        + CLUSTER_WEIGHT * cluster_loss
        + SEPARATION_WEIGHT * separation_loss
        + 1e-4 * l1_loss
    )

    return total_loss, {
        "classification": float(classification_loss.item()),
        "cluster": float(cluster_loss.item()),
        "separation": float(separation_loss.item()),
        "l1": float(l1_loss.item()),
    }


def compute_full_bank_linear_loss(model: ConfidenceWeightedWithLinearHead, logits, distances, targets):
    classification_loss = F.binary_cross_entropy_with_logits(logits, targets)

    positive_proto_mask = torch.matmul(targets, model.base_model.prototype_label_matrix.T) > 0
    negative_proto_mask = torch.matmul(1.0 - targets, model.base_model.prototype_label_matrix.T) > 0

    positive_min_distances, positive_valid_rows = masked_min(distances, positive_proto_mask)
    cluster_loss = positive_min_distances.mean() if positive_valid_rows.any() else torch.tensor(0.0, device=distances.device)

    negative_min_distances, negative_valid_rows = masked_min(distances, negative_proto_mask)
    separation_loss = (
        F.relu(SEPARATION_MARGIN - negative_min_distances).mean()
        if negative_valid_rows.any()
        else torch.tensor(0.0, device=distances.device)
    )

    weight_regularizer, weight_l1, weight_l2 = compute_weight_regularizer(model.base_model)
    off_class_mask = 1.0 - model.prototype_class_matrix
    head_l1_loss = torch.abs(model.linear_head.weight * off_class_mask).mean()

    total_loss = (
        CLASSIFICATION_WEIGHT * classification_loss
        + CLUSTER_WEIGHT * cluster_loss
        + SEPARATION_WEIGHT * separation_loss
        + weight_regularizer
        + POSTHOC_LINEAR_HEAD_L1_WEIGHT * head_l1_loss
    )

    return total_loss, {
        "classification": float(classification_loss.item()),
        "cluster": float(cluster_loss.item()),
        "separation": float(separation_loss.item()),
        "weight_l1": float(weight_l1.item()),
        "weight_l2": float(weight_l2.item()),
        "head_l1": float(head_l1_loss.item()),
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


def run_global_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    stats = {"classification": 0.0, "cluster": 0.0, "separation": 0.0, "weight_l1": 0.0, "weight_l2": 0.0}
    steps = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits, distances, _ = model(x_batch)
        loss, loss_stats = compute_loss(model, logits, distances, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        for key in stats:
            stats[key] += loss_stats[key]
        steps += 1

    if steps == 0:
        return 0.0, stats

    averaged_stats = {key: value / steps for key, value in stats.items()}
    return total_loss / steps, averaged_stats


def run_traditional_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    stats = {"classification": 0.0, "cluster": 0.0, "separation": 0.0, "l1": 0.0}
    steps = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits, distances, _ = model(x_batch)
        loss, loss_stats = compute_traditional_loss(model, logits, distances, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        for key in stats:
            stats[key] += loss_stats[key]
        steps += 1

    if steps == 0:
        return 0.0, stats

    averaged_stats = {key: value / steps for key, value in stats.items()}
    return total_loss / steps, averaged_stats


def run_full_bank_linear_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    stats = {"classification": 0.0, "cluster": 0.0, "separation": 0.0, "weight_l1": 0.0, "weight_l2": 0.0, "head_l1": 0.0}
    steps = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits, distances, _ = model(x_batch)
        loss, loss_stats = compute_full_bank_linear_loss(model, logits, distances, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        for key in stats:
            stats[key] += loss_stats[key]
        steps += 1

    if steps == 0:
        return 0.0, stats

    averaged_stats = {key: value / steps for key, value in stats.items()}
    return total_loss / steps, averaged_stats


def _select_best_position(
    distances_column: torch.Tensor,
    label_counts_chunk: torch.Tensor,
    start_idx: int,
) -> tuple[int, float, int]:
    local_best_distance = torch.min(distances_column)
    close_mask = torch.isclose(
        distances_column,
        local_best_distance,
        atol=PROJECTION_TIE_ATOL,
        rtol=PROJECTION_TIE_RTOL,
    )
    candidate_positions = torch.nonzero(close_mask, as_tuple=False).squeeze(1)
    best_chunk_idx = int(candidate_positions[0].item())
    best_label_count = int(label_counts_chunk[best_chunk_idx].item())
    best_global_idx = start_idx + best_chunk_idx

    for candidate_idx in candidate_positions[1:].tolist():
        candidate_idx = int(candidate_idx)
        candidate_label_count = int(label_counts_chunk[candidate_idx].item())
        candidate_global_idx = start_idx + candidate_idx
        if candidate_label_count < best_label_count:
            best_chunk_idx = candidate_idx
            best_label_count = candidate_label_count
            best_global_idx = candidate_global_idx
            continue
        if candidate_label_count == best_label_count and candidate_global_idx < best_global_idx:
            best_chunk_idx = candidate_idx
            best_label_count = candidate_label_count
            best_global_idx = candidate_global_idx

    return best_chunk_idx, float(local_best_distance.item()), best_label_count


def _is_new_projection_better(
    new_distance: float,
    new_label_count: int,
    new_index: int,
    old_distance: float,
    old_label_count: int,
    old_index: int,
) -> bool:
    if new_distance < old_distance - 1e-12:
        return True
    if abs(new_distance - old_distance) <= 1e-12:
        if new_label_count < old_label_count:
            return True
        if new_label_count == old_label_count and new_index < old_index:
            return True
    return False


def _row_to_label_names(row: torch.Tensor) -> list[str]:
    active_indices = torch.nonzero(row > 0, as_tuple=False).squeeze(1).tolist()
    return [TARGET_LABELS[int(idx)] for idx in active_indices]


def _collapse_label_row_to_single_class(
    label_row: torch.Tensor,
    class_positive_counts: torch.Tensor | np.ndarray | list[int] | None = None,
) -> torch.Tensor:
    active_indices = torch.nonzero(label_row > 0, as_tuple=False).squeeze(1).tolist()
    if not active_indices:
        return label_row

    if class_positive_counts is None:
        primary_class_idx = int(min(active_indices))
    elif isinstance(class_positive_counts, torch.Tensor):
        counts = class_positive_counts.detach().cpu().numpy().astype(int).tolist()
        primary_class_idx = _choose_primary_class([int(idx) for idx in active_indices], counts)
    elif isinstance(class_positive_counts, np.ndarray):
        primary_class_idx = _choose_primary_class([int(idx) for idx in active_indices], class_positive_counts.astype(int).tolist())
    else:
        primary_class_idx = _choose_primary_class([int(idx) for idx in active_indices], [int(item) for item in class_positive_counts])

    collapsed = torch.zeros_like(label_row)
    collapsed[int(primary_class_idx)] = 1.0
    return collapsed


def project_prototypes(
    model,
    x_train,
    y_train,
    device,
    chunk_size,
    update_model=True,
    enforce_single_label: bool = False,
    class_positive_counts: torch.Tensor | np.ndarray | list[int] | None = None,
):
    action_text = "Projecting prototypes to unrestricted nearest real samples"
    if not update_model:
        action_text = "Tracing prototypes to unrestricted nearest real samples"
    if enforce_single_label:
        action_text += " with single-label collapse"
    print(f"\n[Projection] {action_text} ...")
    model.eval()

    num_prototypes = model.prototypes.shape[0]
    num_classes = int(y_train.shape[1])
    best_distances = [float("inf")] * num_prototypes
    best_label_counts = [10**9] * num_prototypes
    best_indices = torch.full((num_prototypes,), -1, dtype=torch.long, device=device)
    best_features = torch.zeros_like(model.prototypes.data)
    best_label_matrix = model.prototype_label_matrix.detach().clone()

    with torch.no_grad():
        for start_idx in range(0, x_train.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, x_train.shape[0])
            x_chunk = x_train[start_idx:end_idx].to(device)
            y_chunk = y_train[start_idx:end_idx].to(device)
            label_counts_chunk = y_chunk.sum(dim=1).long()

            encoded_features = model.encode(x_chunk)
            distances = model.prototype_distances(encoded_features)

            for proto_idx in range(num_prototypes):
                selected_chunk_idx, selected_distance, selected_label_count = _select_best_position(
                    distances[:, proto_idx],
                    label_counts_chunk,
                    start_idx,
                )
                selected_global_idx = start_idx + selected_chunk_idx
                if _is_new_projection_better(
                    selected_distance,
                    selected_label_count,
                    selected_global_idx,
                    best_distances[proto_idx],
                    best_label_counts[proto_idx],
                    int(best_indices[proto_idx].item()) if int(best_indices[proto_idx].item()) >= 0 else 10**12,
                ):
                    best_distances[proto_idx] = selected_distance
                    best_label_counts[proto_idx] = selected_label_count
                    best_indices[proto_idx] = selected_global_idx
                    best_features[proto_idx] = encoded_features[selected_chunk_idx]
                    if selected_label_count > 0:
                        selected_label_row = y_chunk[selected_chunk_idx]
                        if enforce_single_label:
                            selected_label_row = _collapse_label_row_to_single_class(
                                selected_label_row,
                                class_positive_counts=class_positive_counts,
                            )
                        best_label_matrix[proto_idx] = selected_label_row

        valid_mask = best_indices >= 0
        if update_model and valid_mask.any():
            model.prototypes.data[valid_mask] = F.normalize(best_features[valid_mask], p=2, dim=1)
            model.update_prototype_label_matrix(best_label_matrix)

    mapped_count = int((best_indices >= 0).sum().item())
    if update_model:
        print(f"[Projection] Updated {mapped_count}/{num_prototypes} prototypes.")
    else:
        print(f"[Projection] Traced {mapped_count}/{num_prototypes} prototypes.")

    projection_matrix_cpu = best_label_matrix.detach().cpu()
    return {
        "mapped_train_indices": best_indices.detach().cpu(),
        "mapped_label_matrix": projection_matrix_cpu,
        "mapped_positive_label_names": [_row_to_label_names(row) for row in projection_matrix_cpu],
        "mapped_label_counts": projection_matrix_cpu.sum(dim=1).to(dtype=torch.int64).cpu(),
        "num_classes": num_classes,
    }


def project_traditional_prototypes(model, x_train, y_train, device, chunk_size, update_model=True):
    action_text = "Pushing traditional prototypes onto nearest real positive samples"
    if not update_model:
        action_text = "Tracing traditional prototypes to nearest real positive samples"
    print(f"\n[Traditional Projection] {action_text} ...")
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

    mapped_count = int((best_indices >= 0).sum().item())
    if update_model:
        print(f"[Traditional Projection] Updated {mapped_count}/{num_prototypes} prototypes.")
    else:
        print(f"[Traditional Projection] Traced {mapped_count}/{num_prototypes} prototypes.")
    return best_indices.detach().cpu()


def calibrate_prototype_weights(model, dataloader, device, num_epochs, lr):
    print(f"[Calibration] Calibrating prototype weights for {num_epochs} epochs ...")

    for param in model.adapter.parameters():
        param.requires_grad = False
    model.prototypes.requires_grad_(False)
    model.prototype_coefficients.requires_grad_(True)

    optimizer = optim.Adam([model.prototype_coefficients], lr=lr)

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
            weight_regularizer, _weight_l1, _weight_l2 = compute_weight_regularizer(model)
            loss = classification_loss + weight_regularizer
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            steps += 1

        average_loss = epoch_loss / max(steps, 1)
        print(f"  [Calibration] Epoch {epoch}: Loss={average_loss:.4f}")

    for param in model.adapter.parameters():
        param.requires_grad = True
    model.prototypes.requires_grad_(True)
    model.prototype_coefficients.requires_grad_(True)


def fine_tune_traditional_last_layer(model, dataloader, device, num_epochs, lr):
    print(f"[Traditional Convex Opt] Fine-tuning last layer for {num_epochs} epochs ...")

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
            loss = classification_loss + 1e-4 * l1_loss
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            steps += 1

        average_loss = epoch_loss / max(steps, 1)
        print(f"  [Traditional Convex Opt] Epoch {epoch}: Loss={average_loss:.4f}")

    for param in model.adapter.parameters():
        param.requires_grad = True
    model.prototypes.requires_grad_(True)


def calibrate_posthoc_linear_head(model, dataloader, device, num_epochs, lr):
    print(f"[Posthoc Linear Head] Training linear head for {num_epochs} epochs ...")

    for param in model.base_model.adapter.parameters():
        param.requires_grad = False
    model.base_model.prototypes.requires_grad_(False)
    model.base_model.prototype_coefficients.requires_grad_(False)
    for param in model.linear_head.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.linear_head.parameters(), lr=lr)

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
            off_class_mask = 1.0 - model.prototype_class_matrix
            l1_loss = torch.abs(model.linear_head.weight * off_class_mask).mean()
            loss = classification_loss + POSTHOC_LINEAR_HEAD_L1_WEIGHT * l1_loss
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            steps += 1

        average_loss = epoch_loss / max(steps, 1)
        print(f"  [Posthoc Linear Head] Epoch {epoch}: Loss={average_loss:.4f}")

    for param in model.base_model.adapter.parameters():
        param.requires_grad = True
    model.base_model.prototypes.requires_grad_(True)
    model.base_model.prototype_coefficients.requires_grad_(True)


def maybe_save_best(model, val_auc, best_val_auc, best_epoch, epoch, best_model_path):
    if val_auc > best_val_auc:
        torch.save(model.state_dict(), best_model_path)
        print(f"  >>> New best validation AUROC: {best_val_auc:.4f} -> {val_auc:.4f}")
        return val_auc, epoch
    return best_val_auc, best_epoch


def _choose_primary_class(active_indices: list[int], class_positive_counts: list[int] | None) -> int:
    if not active_indices:
        return -1
    if not class_positive_counts:
        return int(min(active_indices))
    return int(min(active_indices, key=lambda idx: (class_positive_counts[idx], idx)))


def build_pruning_summary(model, class_positive_counts: torch.Tensor | np.ndarray | list[int] | None = None) -> dict[str, object]:
    weights = model.prototype_weights().detach().cpu().view(-1)
    abs_weights = torch.abs(weights)
    low_mask = abs_weights < PRUNE_WEIGHT_ABS_THRESHOLD
    payload: dict[str, object] = {
        "prune_weight_abs_threshold": float(PRUNE_WEIGHT_ABS_THRESHOLD),
        "candidate_prune_count": int(low_mask.sum().item()),
        "total_prototype_count": int(len(weights)),
        "candidate_prune_fraction": float(low_mask.float().mean().item()) if len(weights) > 0 else float("nan"),
    }
    if class_positive_counts is not None:
        if isinstance(class_positive_counts, torch.Tensor):
            _counts = class_positive_counts.detach().cpu().numpy().astype(int).tolist()
        elif isinstance(class_positive_counts, np.ndarray):
            _counts = class_positive_counts.astype(int).tolist()
        else:
            _counts = [int(item) for item in class_positive_counts]
        per_class_rows = []
        label_matrix = model.prototype_label_matrix.detach().cpu()
        for class_idx, label_name in enumerate(TARGET_LABELS):
            class_mask = label_matrix[:, class_idx] > 0
            if not bool(class_mask.any()):
                candidate_count = 0
                total_count = 0
            else:
                candidate_count = int((low_mask[class_mask]).sum().item())
                total_count = int(class_mask.sum().item())
            per_class_rows.append(
                {
                    "class_idx": int(class_idx),
                    "class_name": label_name,
                    "candidate_prune_count": candidate_count,
                    "class_prototype_count": total_count,
                }
            )
        payload["per_class"] = per_class_rows
    return payload


def build_prototype_report(model, projection_result, train_idx, class_positive_counts: torch.Tensor | np.ndarray | list[int] | None = None):
    rows = []
    weights = model.prototype_weights().detach().cpu().view(-1)
    mapped_train_indices = projection_result["mapped_train_indices"].tolist()
    mapped_label_matrix = projection_result["mapped_label_matrix"].cpu()
    global_train_idx = train_idx.tolist() if hasattr(train_idx, "tolist") else list(train_idx)

    if class_positive_counts is None:
        class_count_list = None
    elif isinstance(class_positive_counts, torch.Tensor):
        class_count_list = class_positive_counts.detach().cpu().numpy().astype(int).tolist()
    elif isinstance(class_positive_counts, np.ndarray):
        class_count_list = class_positive_counts.astype(int).tolist()
    else:
        class_count_list = [int(item) for item in class_positive_counts]

    for proto_id in range(mapped_label_matrix.shape[0]):
        local_idx = int(mapped_train_indices[proto_id])
        global_idx = int(global_train_idx[local_idx]) if local_idx >= 0 else -1
        active_indices = torch.nonzero(mapped_label_matrix[proto_id] > 0, as_tuple=False).squeeze(1).tolist()
        active_indices = [int(idx) for idx in active_indices]
        active_names = [TARGET_LABELS[idx] for idx in active_indices]
        primary_class_idx = _choose_primary_class(active_indices, class_count_list)
        primary_class_name = TARGET_LABELS[primary_class_idx] if primary_class_idx >= 0 else ""
        rows.append(
            {
                "prototype_id": proto_id,
                "prototype_weight": float(weights[proto_id].item()),
                "prototype_abs_weight": float(abs(weights[proto_id].item())),
                "prototype_weight_sign": int(np.sign(weights[proto_id].item())),
                "candidate_for_pruning": bool(abs(weights[proto_id].item()) < PRUNE_WEIGHT_ABS_THRESHOLD),
                "train_local_index": local_idx,
                "train_global_index": global_idx,
                "label_count": int(len(active_indices)),
                "class_indices": "|".join(str(idx) for idx in active_indices),
                "class_names": "|".join(active_names),
                "primary_class_idx": int(primary_class_idx),
                "primary_class_name": primary_class_name,
            }
        )

    return pd.DataFrame(rows)


def analyze_prototype_activation(model, dataloader, device, class_positive_counts: torch.Tensor | np.ndarray | list[int] | None = None):
    _, _, activations = collect_model_outputs(model, dataloader, device)
    top_proto_indices = torch.argmax(activations, dim=1).numpy()
    counts = np.bincount(top_proto_indices, minlength=activations.shape[1])
    total = counts.sum()

    report = build_prototype_report(
        model,
        {
            "mapped_train_indices": torch.full((len(counts),), -1, dtype=torch.long),
            "mapped_label_matrix": model.prototype_label_matrix.detach().cpu(),
        },
        train_idx=np.arange(len(counts)),
        class_positive_counts=class_positive_counts,
    )
    report["activation_count"] = counts
    report["activation_rate"] = [(count / total) if total > 0 else 0.0 for count in counts.tolist()]
    return report.sort_values(by=["activation_count", "prototype_id"], ascending=[False, True]).reset_index(drop=True)


def print_prototype_report(report_df, title):
    print(f"\n=== {title} ===")
    print(f"{'Proto ID':<10} | {'Weight':<11} | {'Primary Class':<35} | {'Labels'}")
    print("-" * 120)
    for row in report_df.itertuples(index=False):
        print(
            f"{row.prototype_id:<10} | {row.prototype_weight:<11.4f} | "
            f"{str(row.primary_class_name):<35} | {row.class_names}"
        )


def print_activation_report(activation_df, title, top_k=20):
    print(f"\n=== {title} ===")
    print(f"{'Proto ID':<10} | {'Weight':<11} | {'Count':<10} | {'Primary Class':<35} | {'Labels'}")
    print("-" * 140)
    for row in activation_df.head(top_k).itertuples(index=False):
        print(
            f"{row.prototype_id:<10} | {row.prototype_weight:<11.4f} | "
            f"{row.activation_count:<10} | {str(row.primary_class_name):<35} | {row.class_names}"
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
        weights = model.prototype_weights().detach().cpu().view(-1).numpy()

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
    sample_labels = torch.argmax(y_sample, dim=1).cpu().numpy()
    proto_primary = model.prototype_primary_label.detach().cpu().numpy()
    proto_sizes = 60.0 + 120.0 * np.clip(np.abs(weights), 0.0, 1.0)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(
        sample_2d[:, 0],
        sample_2d[:, 1],
        c=sample_labels,
        cmap="tab20",
        alpha=0.18,
        s=8,
        label="Reference samples",
    )
    ax.scatter(
        proto_2d[:, 0],
        proto_2d[:, 1],
        c="white",
        edgecolors="black",
        linewidths=0.8,
        s=140,
        marker="X",
        label="Prototypes",
    )

    for proto_id, (x_coord, y_coord) in enumerate(proto_2d):
        ax.text(
            x_coord,
            y_coord,
            f"P{proto_id}\n{weights[proto_id]:+.2f}",
            fontsize=7,
            ha="center",
            va="bottom",
        )

    ax.set_title("Signed-weight ProtoPNet Prototypes in Feature Space (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    primary_class_path = str(Path(output_path).with_name(Path(output_path).stem + "_primary_class.png"))
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(
        sample_2d[:, 0],
        sample_2d[:, 1],
        c=sample_labels,
        cmap="tab20",
        alpha=0.12,
        s=8,
        label="Reference samples",
    )
    scatter = ax.scatter(
        proto_2d[:, 0],
        proto_2d[:, 1],
        c=proto_primary,
        cmap="tab20",
        edgecolors="black",
        linewidths=0.7,
        s=proto_sizes,
        marker="X",
        alpha=0.9,
        label="Prototypes (primary class)",
    )
    for proto_id, (x_coord, y_coord) in enumerate(proto_2d):
        ax.text(
            x_coord,
            y_coord,
            f"P{proto_id}",
            fontsize=6,
            ha="center",
            va="bottom",
            alpha=0.85,
        )
    ax.set_title("Signed-weight ProtoPNet Prototypes by Primary Class (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(primary_class_path, dpi=220)
    plt.close(fig)

    negative_white_path = str(Path(output_path).with_name(Path(output_path).stem + "_negative_white.png"))
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(
        sample_2d[:, 0],
        sample_2d[:, 1],
        c=sample_labels,
        cmap="tab20",
        alpha=0.10,
        s=8,
        label="Reference samples",
    )

    positive_mask = weights > PRUNE_WEIGHT_ABS_THRESHOLD
    negative_mask = weights < -PRUNE_WEIGHT_ABS_THRESHOLD
    near_zero_mask = np.abs(weights) <= PRUNE_WEIGHT_ABS_THRESHOLD

    if np.any(positive_mask):
        ax.scatter(
            proto_2d[positive_mask, 0],
            proto_2d[positive_mask, 1],
            c=proto_primary[positive_mask],
            cmap="tab20",
            edgecolors="black",
            linewidths=0.7,
            s=proto_sizes[positive_mask],
            marker="X",
            alpha=0.92,
            label="Positive-weight prototypes",
        )
    if np.any(negative_mask):
        ax.scatter(
            proto_2d[negative_mask, 0],
            proto_2d[negative_mask, 1],
            c="white",
            edgecolors="black",
            linewidths=0.9,
            s=proto_sizes[negative_mask],
            marker="X",
            alpha=0.98,
            label="Negative-weight prototypes",
        )
    if np.any(near_zero_mask):
        ax.scatter(
            proto_2d[near_zero_mask, 0],
            proto_2d[near_zero_mask, 1],
            c="lightgray",
            edgecolors="dimgray",
            linewidths=0.7,
            s=proto_sizes[near_zero_mask],
            marker="X",
            alpha=0.9,
            label="Near-zero prototypes",
        )

    for proto_id, (x_coord, y_coord) in enumerate(proto_2d):
        ax.text(
            x_coord,
            y_coord,
            f"P{proto_id}",
            fontsize=6,
            ha="center",
            va="bottom",
            alpha=0.82,
        )

    ax.set_title("Signed-weight ProtoPNet: Negative Prototypes in White (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(negative_white_path, dpi=220)
    plt.close(fig)
    print("[t-SNE] Done.")


def export_traditional_prototype_tsne(model, x_reference, y_reference, device, output_path, sample_size):
    print(f"\n[t-SNE] Exporting traditional prototype visualization to {output_path} ...")

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
        weights = model.prototype_onclass_weights().detach().cpu().view(-1).numpy()
        proto_labels = model.prototype_labels.detach().cpu().numpy()

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
    sample_labels = torch.argmax(y_sample, dim=1).cpu().numpy()
    proto_sizes = 60.0 + 120.0 * np.clip(np.abs(weights), 0.0, 1.0)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(
        sample_2d[:, 0],
        sample_2d[:, 1],
        c=sample_labels,
        cmap="tab20",
        alpha=0.12,
        s=8,
        label="Reference samples",
    )
    ax.scatter(
        proto_2d[:, 0],
        proto_2d[:, 1],
        c=proto_labels,
        cmap="tab20",
        edgecolors="black",
        linewidths=0.8,
        s=proto_sizes,
        marker="X",
        alpha=0.92,
        label="Traditional prototypes",
    )
    for proto_id, (x_coord, y_coord) in enumerate(proto_2d):
        ax.text(
            x_coord,
            y_coord,
            f"P{proto_id}\n{weights[proto_id]:+.2f}",
            fontsize=6,
            ha="center",
            va="bottom",
        )
    ax.set_title("Traditional ProtoPNet Prototypes by On-class Last-layer Weight (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    negative_white_path = str(Path(output_path).with_name(Path(output_path).stem + "_negative_white.png"))
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(
        sample_2d[:, 0],
        sample_2d[:, 1],
        c=sample_labels,
        cmap="tab20",
        alpha=0.10,
        s=8,
        label="Reference samples",
    )
    positive_mask = weights > PRUNE_WEIGHT_ABS_THRESHOLD
    negative_mask = weights < -PRUNE_WEIGHT_ABS_THRESHOLD
    near_zero_mask = np.abs(weights) <= PRUNE_WEIGHT_ABS_THRESHOLD
    if np.any(positive_mask):
        ax.scatter(
            proto_2d[positive_mask, 0],
            proto_2d[positive_mask, 1],
            c=proto_labels[positive_mask],
            cmap="tab20",
            edgecolors="black",
            linewidths=0.8,
            s=proto_sizes[positive_mask],
            marker="X",
            alpha=0.92,
            label="Positive on-class weight",
        )
    if np.any(negative_mask):
        ax.scatter(
            proto_2d[negative_mask, 0],
            proto_2d[negative_mask, 1],
            c="white",
            edgecolors="black",
            linewidths=0.9,
            s=proto_sizes[negative_mask],
            marker="X",
            alpha=0.98,
            label="Negative on-class weight",
        )
    if np.any(near_zero_mask):
        ax.scatter(
            proto_2d[near_zero_mask, 0],
            proto_2d[near_zero_mask, 1],
            c="lightgray",
            edgecolors="dimgray",
            linewidths=0.8,
            s=proto_sizes[near_zero_mask],
            marker="X",
            alpha=0.92,
            label="Near-zero on-class weight",
        )
    for proto_id, (x_coord, y_coord) in enumerate(proto_2d):
        ax.text(
            x_coord,
            y_coord,
            f"P{proto_id}",
            fontsize=6,
            ha="center",
            va="bottom",
            alpha=0.82,
        )
    ax.set_title("Traditional ProtoPNet: Negative On-class Weights in White (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(negative_white_path, dpi=220)
    plt.close(fig)
    print("[t-SNE] Done.")
