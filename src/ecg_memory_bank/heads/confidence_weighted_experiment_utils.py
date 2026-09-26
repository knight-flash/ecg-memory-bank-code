#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prototype branch training utilities.

Purpose
-------
Share three-branch training, checkpoint restoration, score evaluation, and
manifest export across paper prototype tasks.

Method
------
Train traditional, signed single-label, and full-bank linear aggregators on
identical prepared datasets, then collect AUROC and case-link artifacts.

Inputs
------
Prepared ECG embeddings, binary label matrices, class prototype plan, loaders,
optimizer/training options.

Outputs
-------
External branch checkpoints, per-class/aggregate metrics, prototype reports,
and CI inputs.

Paper parameters
----------------
Formal wrappers provide budget, learning schedule, seed and bootstrap count;
helper functions do not choose the study cohort.

Shapes and meaning
------------------
Vectors are [N,D], labels [N,C], prototype responses [N,P], and branch scores
[N,C].
"""


from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score
from ci_bootstrap_utils import build_multilabel_ci_report, save_json as save_ci_json, write_multilabel_ci_outputs


def save_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_multilabel_benchmark_compatible(
    base_module,
    y_true: torch.Tensor,
    y_logits: torch.Tensor,
    label_names: list[str],
    title: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_scores = torch.sigmoid(y_logits).detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    print(f"\n=== {title} ===")
    print(f"{'Class':<45} | {'AUROC':<10} | {'Pos':<6} | {'Neg'}")
    print("-" * 90)

    per_class_macro_values: list[float] = []
    valid_class_count = 0
    rows: list[dict[str, Any]] = []
    num_samples = int(y_true_np.shape[0])

    for class_idx, label_name in enumerate(label_names):
        positive_count = int(np.sum(y_true_np[:, class_idx]))
        negative_count = int(num_samples - positive_count)
        eligible = positive_count > 0 and negative_count > 0
        auc_value: float | None = None
        status_text = "N/A"

        if eligible:
            try:
                auc_value = float(base_module.roc_auc_score(y_true_np[:, class_idx], y_scores[:, class_idx]))
            except Exception:
                auc_value = float("nan")
            if auc_value == auc_value:
                valid_class_count += 1
                per_class_macro_values.append(auc_value)
                status_text = f"{auc_value:.4f}"
            else:
                per_class_macro_values.append(0.5)
        else:
            per_class_macro_values.append(0.5)

        print(f"{label_name:<45} | {status_text:<10} | {positive_count:<6} | {negative_count}")
        rows.append(
            {
                "class_name": label_name,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "eligible_for_auc": bool(eligible),
                "auroc": auc_value,
            }
        )

    try:
        micro_auc = float(base_module.roc_auc_score(y_true_np.ravel(), y_scores.ravel()))
    except Exception:
        micro_auc = float("nan")

    macro_auc = float(np.mean(per_class_macro_values)) if per_class_macro_values else float("nan")
    skipped_class_count = int(len(label_names) - valid_class_count)

    print("-" * 90)
    print(f"[{title}] Macro AUROC (benchmark-compatible): {macro_auc:.4f}")
    print(f"[{title}] Micro AUROC (benchmark-compatible): {micro_auc:.4f}")

    return (
        {
            "macro_auroc": macro_auc,
            "micro_auroc": micro_auc,
            "valid_class_count": int(valid_class_count),
            "skipped_class_count": skipped_class_count,
            "macro_fill_value_for_ineligible_auc": 0.5,
            "aggregation_level": "record_level_single_embedding",
        },
        pd.DataFrame(rows, columns=["class_name", "positive_count", "negative_count", "eligible_for_auc", "auroc"]),
    )


def write_label_manifest(
    path: str | Path,
    *,
    task_payload: dict[str, Any],
    label_selection: dict[str, Any],
    split_counts: dict[str, int],
    extra_payload: dict[str, Any] | None = None,
) -> None:
    payload = {
        "task_name": task_payload["task_name"],
        "label_selection": label_selection,
        "labels": task_payload["manifest_rows"],
        "split_counts": split_counts,
    }
    if extra_payload:
        payload.update(extra_payload)
    save_json(path, payload)


def write_prototype_plan(
    path: str | Path,
    *,
    manifest_rows: list[dict[str, Any]],
    prototype_counts: list[int],
    y_train: torch.Tensor,
) -> None:
    positive_counts = y_train.sum(dim=0).cpu().numpy().astype(int).tolist()
    rows = []
    for item, proto_count, positive_count in zip(manifest_rows, prototype_counts, positive_counts):
        rows.append(
            {
                "class_idx": item["class_idx"],
                "canonical_label": item["canonical_label"],
                "source_column": item["source_column"],
                "group": item["group"],
                "alias_count": item["alias_count"],
                "fixed_prototype_count": int(proto_count),
                "train_positive_count": int(positive_count),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize_split_counts(**counts: int) -> dict[str, int]:
    return {key: int(value) for key, value in counts.items()}


def train_three_branch_multilabel_experiment(
    *,
    base_module,
    task_name: str,
    output_root: Path,
    checkpoint_root: Path,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    train_idx_original,
    device: torch.device,
    max_epochs: int,
    label_selection: dict[str, Any],
    label_stats_rows: list[dict[str, Any]],
    mapping_warnings: list[str] | None,
    split_protocol: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    prototype_counts: list[int],
    summary_json_path: Path,
    primary_per_class_csv_path: Path,
    main_test_name: str = "test",
    primary_eval_split_name: str | None = None,
    extra_test_sets: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    extra_summary_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_loader = base_module.make_dataloader(x_train, y_train, base_module.TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = base_module.make_dataloader(x_val, y_val, base_module.EVAL_BATCH_SIZE, shuffle=False)
    test_loader = base_module.make_dataloader(x_test, y_test, base_module.EVAL_BATCH_SIZE, shuffle=False)
    extra_test_loaders = {}
    if extra_test_sets:
        for split_name, (x_extra, y_extra) in extra_test_sets.items():
            extra_test_loaders[split_name] = (
                base_module.make_dataloader(x_extra, y_extra, base_module.EVAL_BATCH_SIZE, shuffle=False),
                y_extra,
            )

    init_prototypes, prototype_label_matrix = base_module.initialize_fixed_prototypes(
        x_train,
        y_train,
        prototype_counts,
        base_module.INIT_MAX_SAMPLES_PER_CLASS,
    )

    traditional_model = base_module.TraditionalProtoPNetComparator(
        init_prototypes,
        torch.argmax(prototype_label_matrix, dim=1).long(),
        len(base_module.TARGET_LABELS),
    ).to(device)
    single_label_model = base_module.ConfidenceWeightedProtoPNet(
        init_prototypes,
        prototype_label_matrix,
        len(base_module.TARGET_LABELS),
    ).to(device)
    full_bank_linear_model = base_module.ConfidenceWeightedWithLinearHead(
        base_module.ConfidenceWeightedProtoPNet(
            init_prototypes,
            prototype_label_matrix,
            len(base_module.TARGET_LABELS),
        ).to(device)
    ).to(device)

    traditional_best_model_path = str(checkpoint_root / f"{task_name}_traditional_best.pt")
    traditional_last_model_path = str(checkpoint_root / f"{task_name}_traditional_last.pt")
    traditional_best_snapshot_path = str(checkpoint_root / f"{task_name}_traditional_best_snapshot.pt")
    traditional_best_per_class_csv = output_root / "per_class_auc_traditional_best.csv"
    traditional_best_tsne_png = str(output_root / f"{task_name}_traditional_best_tsne.png")
    traditional_prototype_report_csv = output_root / f"{task_name}_traditional_prototype_report.csv"

    single_label_best_model_path = str(checkpoint_root / f"{task_name}_single_label_best.pt")
    single_label_last_model_path = str(checkpoint_root / f"{task_name}_single_label_last.pt")
    single_label_best_snapshot_path = str(checkpoint_root / f"{task_name}_single_label_best_snapshot.pt")
    single_label_best_per_class_csv = output_root / "per_class_auc_single_label_best.csv"
    single_label_prototype_report_csv = output_root / f"{task_name}_single_label_prototype_report.csv"
    single_label_tsne_png = str(output_root / f"{task_name}_single_label_tsne.png")

    full_bank_linear_best_model_path = str(checkpoint_root / f"{task_name}_full_bank_linear_best.pt")
    full_bank_linear_last_model_path = str(checkpoint_root / f"{task_name}_full_bank_linear_last.pt")
    full_bank_linear_best_snapshot_path = str(checkpoint_root / f"{task_name}_full_bank_linear_best_snapshot.pt")
    full_bank_linear_best_per_class_csv = output_root / "per_class_auc_full_bank_linear_best.csv"
    full_bank_linear_prototype_report_csv = output_root / f"{task_name}_full_bank_linear_prototype_report.csv"
    full_bank_linear_tsne_png = str(output_root / f"{task_name}_full_bank_linear_tsne.png")

    traditional_optimizer = torch.optim.Adam(traditional_model.parameters(), lr=base_module.GLOBAL_LR)
    single_label_optimizer = torch.optim.Adam(single_label_model.parameters(), lr=base_module.GLOBAL_LR)
    full_bank_linear_optimizer = torch.optim.Adam(full_bank_linear_model.parameters(), lr=base_module.GLOBAL_LR)

    traditional_best_val_auc = 0.0
    traditional_best_epoch = 0
    single_label_best_val_auc = 0.0
    single_label_best_epoch = 0
    full_bank_linear_best_val_auc = 0.0
    full_bank_linear_best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        traditional_average_loss, traditional_loss_stats = base_module.run_traditional_epoch(
            traditional_model,
            train_loader,
            traditional_optimizer,
            device,
        )
        print(
            f"[Traditional] Epoch {epoch}: Loss={traditional_average_loss:.4f} | "
            f"cls={traditional_loss_stats['classification']:.4f} | "
            f"cluster={traditional_loss_stats['cluster']:.4f} | "
            f"sep={traditional_loss_stats['separation']:.4f} | "
            f"l1={traditional_loss_stats['l1']:.4f}"
        )

        single_label_average_loss, single_label_loss_stats = base_module.run_global_epoch(
            single_label_model, train_loader, single_label_optimizer, device
        )
        print(
            f"[SingleLabel] Epoch {epoch}: Loss={single_label_average_loss:.4f} | "
            f"cls={single_label_loss_stats['classification']:.4f} | "
            f"cluster={single_label_loss_stats['cluster']:.4f} | "
            f"sep={single_label_loss_stats['separation']:.4f} | "
            f"w_l1={single_label_loss_stats['weight_l1']:.4f} | "
            f"w_l2={single_label_loss_stats['weight_l2']:.4f}"
        )

        full_bank_linear_average_loss, full_bank_linear_loss_stats = base_module.run_full_bank_linear_epoch(
            full_bank_linear_model,
            train_loader,
            full_bank_linear_optimizer,
            device,
        )
        print(
            f"[FullBankLinear] Epoch {epoch}: Loss={full_bank_linear_average_loss:.4f} | "
            f"cls={full_bank_linear_loss_stats['classification']:.4f} | "
            f"cluster={full_bank_linear_loss_stats['cluster']:.4f} | "
            f"sep={full_bank_linear_loss_stats['separation']:.4f} | "
            f"w_l1={full_bank_linear_loss_stats['weight_l1']:.4f} | "
            f"w_l2={full_bank_linear_loss_stats['weight_l2']:.4f} | "
            f"head_l1={full_bank_linear_loss_stats['head_l1']:.4f}"
        )

        if epoch % base_module.EVAL_INTERVAL == 0:
            traditional_val_metrics, _ = evaluate_multilabel_benchmark_compatible(
                base_module,
                y_val,
                base_module.collect_logits(traditional_model, val_loader, device),
                base_module.TARGET_LABELS,
                f"Traditional Epoch {epoch} Val",
            )
            traditional_best_val_auc, traditional_best_epoch = base_module.maybe_save_best(
                traditional_model,
                traditional_val_metrics["macro_auroc"],
                traditional_best_val_auc,
                traditional_best_epoch,
                epoch,
                traditional_best_model_path,
            )

            single_label_val_metrics, _ = evaluate_multilabel_benchmark_compatible(
                base_module,
                y_val,
                base_module.collect_logits(single_label_model, val_loader, device),
                base_module.TARGET_LABELS,
                f"Single-Label Epoch {epoch} Val",
            )
            single_label_best_val_auc, single_label_best_epoch = base_module.maybe_save_best(
                single_label_model,
                single_label_val_metrics["macro_auroc"],
                single_label_best_val_auc,
                single_label_best_epoch,
                epoch,
                single_label_best_model_path,
            )

            full_bank_linear_val_metrics, _ = evaluate_multilabel_benchmark_compatible(
                base_module,
                y_val,
                base_module.collect_logits(full_bank_linear_model, val_loader, device),
                base_module.TARGET_LABELS,
                f"Full-Bank Linear Epoch {epoch} Val",
            )
            full_bank_linear_best_val_auc, full_bank_linear_best_epoch = base_module.maybe_save_best(
                full_bank_linear_model,
                full_bank_linear_val_metrics["macro_auroc"],
                full_bank_linear_best_val_auc,
                full_bank_linear_best_epoch,
                epoch,
                full_bank_linear_best_model_path,
            )

        if epoch % base_module.PROJECTION_INTERVAL == 0:
            base_module.project_traditional_prototypes(
                traditional_model,
                x_train,
                y_train,
                device,
                base_module.EVAL_BATCH_SIZE,
                update_model=True,
            )
            traditional_val_metrics_after_proj, _ = evaluate_multilabel_benchmark_compatible(
                base_module,
                y_val,
                base_module.collect_logits(traditional_model, val_loader, device),
                base_module.TARGET_LABELS,
                f"Traditional Epoch {epoch} Val After Projection",
            )
            traditional_best_val_auc, traditional_best_epoch = base_module.maybe_save_best(
                traditional_model,
                traditional_val_metrics_after_proj["macro_auroc"],
                traditional_best_val_auc,
                traditional_best_epoch,
                epoch,
                traditional_best_model_path,
            )
            base_module.fine_tune_traditional_last_layer(
                traditional_model,
                train_loader,
                device,
                base_module.LAST_LAYER_FINE_TUNE_EPOCHS,
                base_module.LAST_LAYER_LR,
            )
            traditional_val_metrics_after_convex, _ = evaluate_multilabel_benchmark_compatible(
                base_module,
                y_val,
                base_module.collect_logits(traditional_model, val_loader, device),
                base_module.TARGET_LABELS,
                f"Traditional Epoch {epoch} Val After Convex Opt",
            )
            traditional_best_val_auc, traditional_best_epoch = base_module.maybe_save_best(
                traditional_model,
                traditional_val_metrics_after_convex["macro_auroc"],
                traditional_best_val_auc,
                traditional_best_epoch,
                epoch,
                traditional_best_model_path,
            )
            traditional_optimizer = torch.optim.Adam(traditional_model.parameters(), lr=base_module.GLOBAL_LR)

            base_module.project_prototypes(
                single_label_model,
                x_train,
                y_train,
                device,
                base_module.EVAL_BATCH_SIZE,
                update_model=True,
                enforce_single_label=True,
                class_positive_counts=y_train.sum(dim=0),
            )
            single_label_val_metrics_after_proj, _ = evaluate_multilabel_benchmark_compatible(
                base_module,
                y_val,
                base_module.collect_logits(single_label_model, val_loader, device),
                base_module.TARGET_LABELS,
                f"Single-Label Epoch {epoch} Val After Projection",
            )
            single_label_best_val_auc, single_label_best_epoch = base_module.maybe_save_best(
                single_label_model,
                single_label_val_metrics_after_proj["macro_auroc"],
                single_label_best_val_auc,
                single_label_best_epoch,
                epoch,
                single_label_best_model_path,
            )
            base_module.calibrate_prototype_weights(
                single_label_model,
                train_loader,
                device,
                base_module.LAST_LAYER_FINE_TUNE_EPOCHS,
                base_module.LAST_LAYER_LR,
            )
            single_label_val_metrics_after_calibration, _ = evaluate_multilabel_benchmark_compatible(
                base_module,
                y_val,
                base_module.collect_logits(single_label_model, val_loader, device),
                base_module.TARGET_LABELS,
                f"Single-Label Epoch {epoch} Val After Prototype Weight Calibration",
            )
            single_label_best_val_auc, single_label_best_epoch = base_module.maybe_save_best(
                single_label_model,
                single_label_val_metrics_after_calibration["macro_auroc"],
                single_label_best_val_auc,
                single_label_best_epoch,
                epoch,
                single_label_best_model_path,
            )
            single_label_optimizer = torch.optim.Adam(single_label_model.parameters(), lr=base_module.GLOBAL_LR)

            base_module.project_prototypes(
                full_bank_linear_model.base_model,
                x_train,
                y_train,
                device,
                base_module.EVAL_BATCH_SIZE,
                update_model=True,
            )
            full_bank_linear_val_metrics_after_proj, _ = evaluate_multilabel_benchmark_compatible(
                base_module,
                y_val,
                base_module.collect_logits(full_bank_linear_model, val_loader, device),
                base_module.TARGET_LABELS,
                f"Full-Bank Linear Epoch {epoch} Val After Projection",
            )
            full_bank_linear_best_val_auc, full_bank_linear_best_epoch = base_module.maybe_save_best(
                full_bank_linear_model,
                full_bank_linear_val_metrics_after_proj["macro_auroc"],
                full_bank_linear_best_val_auc,
                full_bank_linear_best_epoch,
                epoch,
                full_bank_linear_best_model_path,
            )
            base_module.calibrate_prototype_weights(
                full_bank_linear_model.base_model,
                train_loader,
                device,
                base_module.LAST_LAYER_FINE_TUNE_EPOCHS,
                base_module.LAST_LAYER_LR,
            )
            full_bank_linear_optimizer = torch.optim.Adam(full_bank_linear_model.parameters(), lr=base_module.GLOBAL_LR)

    with torch.no_grad():
        traditional_last_logits = base_module.collect_logits(traditional_model, test_loader, device)
        traditional_last_metrics, traditional_last_df = evaluate_multilabel_benchmark_compatible(
            base_module,
            y_test,
            traditional_last_logits,
            base_module.TARGET_LABELS,
            f"Traditional Last Model on {main_test_name}",
        )
    torch.save(traditional_model.state_dict(), traditional_last_model_path)

    traditional_best_metrics = dict(traditional_last_metrics)
    traditional_best_df = traditional_last_df
    if Path(traditional_best_model_path).exists():
        traditional_model.load_state_dict(torch.load(traditional_best_model_path, map_location=device))
        with torch.no_grad():
            traditional_best_logits = base_module.collect_logits(traditional_model, test_loader, device)
            traditional_best_metrics, traditional_best_df = evaluate_multilabel_benchmark_compatible(
                base_module,
                y_test,
                traditional_best_logits,
                base_module.TARGET_LABELS,
                f"Traditional Best Model on {main_test_name}",
            )
        torch.save(traditional_model.state_dict(), traditional_best_snapshot_path)
    traditional_best_df.to_csv(traditional_best_per_class_csv, index=False)

    traditional_extra_metrics: dict[str, Any] = {}
    for split_name, (extra_loader, extra_y) in extra_test_loaders.items():
        traditional_extra_metrics[split_name], _ = evaluate_multilabel_benchmark_compatible(
            base_module,
            extra_y,
            base_module.collect_logits(traditional_model, extra_loader, device),
            base_module.TARGET_LABELS,
            f"Traditional Best Model on {split_name}",
        )

    with torch.no_grad():
        single_label_last_logits = base_module.collect_logits(single_label_model, test_loader, device)
        single_label_last_metrics, single_label_last_df = evaluate_multilabel_benchmark_compatible(
            base_module,
            y_test,
            single_label_last_logits,
            base_module.TARGET_LABELS,
            f"Single-Label Last Model on {main_test_name}",
        )
    torch.save(single_label_model.state_dict(), single_label_last_model_path)

    single_label_best_metrics = dict(single_label_last_metrics)
    single_label_best_df = single_label_last_df
    if Path(single_label_best_model_path).exists():
        single_label_model.load_state_dict(torch.load(single_label_best_model_path, map_location=device))
        with torch.no_grad():
            single_label_best_logits = base_module.collect_logits(single_label_model, test_loader, device)
            single_label_best_metrics, single_label_best_df = evaluate_multilabel_benchmark_compatible(
                base_module,
                y_test,
                single_label_best_logits,
                base_module.TARGET_LABELS,
                f"Single-Label Best Model on {main_test_name}",
            )
        torch.save(single_label_model.state_dict(), single_label_best_snapshot_path)
    single_label_best_df.to_csv(single_label_best_per_class_csv, index=False)

    single_label_extra_metrics: dict[str, Any] = {}
    for split_name, (extra_loader, extra_y) in extra_test_loaders.items():
        single_label_extra_metrics[split_name], _ = evaluate_multilabel_benchmark_compatible(
            base_module,
            extra_y,
            base_module.collect_logits(single_label_model, extra_loader, device),
            base_module.TARGET_LABELS,
            f"Single-Label Best Model on {split_name}",
        )

    with torch.no_grad():
        full_bank_linear_last_logits = base_module.collect_logits(full_bank_linear_model, test_loader, device)
        full_bank_linear_last_metrics, full_bank_linear_last_df = evaluate_multilabel_benchmark_compatible(
            base_module,
            y_test,
            full_bank_linear_last_logits,
            base_module.TARGET_LABELS,
            f"Full-Bank Linear Last Model on {main_test_name}",
        )
    torch.save(full_bank_linear_model.state_dict(), full_bank_linear_last_model_path)

    full_bank_linear_best_metrics = dict(full_bank_linear_last_metrics)
    full_bank_linear_best_df = full_bank_linear_last_df
    if Path(full_bank_linear_best_model_path).exists():
        full_bank_linear_model.load_state_dict(torch.load(full_bank_linear_best_model_path, map_location=device))
        with torch.no_grad():
            full_bank_linear_best_logits = base_module.collect_logits(full_bank_linear_model, test_loader, device)
            full_bank_linear_best_metrics, full_bank_linear_best_df = evaluate_multilabel_benchmark_compatible(
                base_module,
                y_test,
                full_bank_linear_best_logits,
                base_module.TARGET_LABELS,
                f"Full-Bank Linear Best Model on {main_test_name}",
            )
        torch.save(full_bank_linear_model.state_dict(), full_bank_linear_best_snapshot_path)
    full_bank_linear_best_df.to_csv(full_bank_linear_best_per_class_csv, index=False)

    full_bank_linear_extra_metrics: dict[str, Any] = {}
    full_bank_linear_extra_dfs: dict[str, pd.DataFrame] = {}
    for split_name, (extra_loader, extra_y) in extra_test_loaders.items():
        full_bank_linear_extra_metrics[split_name], full_bank_linear_extra_dfs[split_name] = evaluate_multilabel_benchmark_compatible(
            base_module,
            extra_y,
            base_module.collect_logits(full_bank_linear_model, extra_loader, device),
            base_module.TARGET_LABELS,
            f"Full-Bank Linear Best Model on {split_name}",
        )

    traditional_projection = base_module.project_traditional_prototypes(
        traditional_model, x_train, y_train, device, base_module.EVAL_BATCH_SIZE, update_model=False
    )
    traditional_report_df = pd.DataFrame(
        {
            "prototype_id": list(range(len(traditional_projection))),
            "train_local_index": traditional_projection.tolist(),
        }
    )
    traditional_report_df.to_csv(traditional_prototype_report_csv, index=False)
    base_module.export_traditional_prototype_tsne(
        traditional_model, x_train, y_train, device, traditional_best_tsne_png, base_module.TSNE_SAMPLE_SIZE
    )

    single_label_projection = base_module.project_prototypes(
        single_label_model,
        x_train,
        y_train,
        device,
        base_module.EVAL_BATCH_SIZE,
        update_model=False,
        enforce_single_label=True,
        class_positive_counts=y_train.sum(dim=0),
    )
    single_label_report_df = base_module.build_prototype_report(
        single_label_model,
        single_label_projection,
        train_idx_original,
        class_positive_counts=y_train.sum(dim=0),
    )
    single_label_report_df.to_csv(single_label_prototype_report_csv, index=False)
    base_module.export_prototype_tsne(
        single_label_model, x_train, y_train, device, single_label_tsne_png, base_module.TSNE_SAMPLE_SIZE
    )

    full_bank_projection = base_module.project_prototypes(
        full_bank_linear_model.base_model,
        x_train,
        y_train,
        device,
        base_module.EVAL_BATCH_SIZE,
        update_model=False,
    )
    full_bank_report_df = base_module.build_prototype_report(
        full_bank_linear_model.base_model,
        full_bank_projection,
        train_idx_original,
        class_positive_counts=y_train.sum(dim=0),
    )
    full_bank_report_df.to_csv(full_bank_linear_prototype_report_csv, index=False)
    base_module.export_prototype_tsne(
        full_bank_linear_model.base_model, x_train, y_train, device, full_bank_linear_tsne_png, base_module.TSNE_SAMPLE_SIZE
    )

    primary_per_class_csv_path.parent.mkdir(parents=True, exist_ok=True)
    eval_split_name = primary_eval_split_name or main_test_name
    if eval_split_name == main_test_name:
        full_bank_linear_best_df.to_csv(primary_per_class_csv_path, index=False)
    elif eval_split_name in full_bank_linear_extra_dfs:
        full_bank_linear_extra_dfs[eval_split_name].to_csv(primary_per_class_csv_path, index=False)
    else:
        full_bank_linear_best_df.to_csv(primary_per_class_csv_path, index=False)

    traditional_summary_metrics: Any = traditional_best_metrics
    single_label_summary_metrics: Any = single_label_best_metrics
    full_bank_linear_summary_metrics: Any = full_bank_linear_best_metrics
    if extra_test_loaders or main_test_name != "test":
        traditional_summary_metrics = {main_test_name: traditional_best_metrics, **traditional_extra_metrics}
        single_label_summary_metrics = {main_test_name: single_label_best_metrics, **single_label_extra_metrics}
        full_bank_linear_summary_metrics = {main_test_name: full_bank_linear_best_metrics, **full_bank_linear_extra_metrics}

    summary_payload = {
        "experiment_name": task_name,
        "model_family": "signed_weight_retrieval_protopnet",
        "retained_branches": [
            "traditional_best",
            "single_label_best",
            "full_bank_linear_best",
        ],
        "split_protocol": split_protocol,
        "mapping_warnings": list(mapping_warnings or []),
        "label_stats": label_stats_rows,
        "label_count": len(base_module.TARGET_LABELS),
        "label_selection": label_selection,
        "split": {
            "train_count": int(len(x_train)),
            "val_count": int(len(x_val)),
            "test_count": int(len(x_test)),
        },
        "metrics": {
            "traditional_best": traditional_summary_metrics,
            "single_label_best": single_label_summary_metrics,
            "full_bank_linear_best": full_bank_linear_summary_metrics,
        },
        "artifacts": {
            "summary_json": str(summary_json_path),
            "target_label_manifest_json": str(output_root / "target_label_manifest.json"),
            "prototype_count_plan_csv": str(output_root / "prototype_count_plan.csv"),
            "traditional_best_checkpoint": traditional_best_model_path,
            "traditional_best_snapshot_checkpoint": traditional_best_snapshot_path,
            "traditional_last_checkpoint": traditional_last_model_path,
            "traditional_best_per_class_csv": str(traditional_best_per_class_csv),
            "traditional_best_tsne_png": traditional_best_tsne_png,
            "traditional_prototype_report_csv": str(traditional_prototype_report_csv),
            "single_label_best_checkpoint": single_label_best_model_path,
            "single_label_best_snapshot_checkpoint": single_label_best_snapshot_path,
            "single_label_last_checkpoint": single_label_last_model_path,
            "single_label_best_per_class_csv": str(single_label_best_per_class_csv),
            "single_label_tsne_png": single_label_tsne_png,
            "single_label_prototype_report_csv": str(single_label_prototype_report_csv),
            "full_bank_linear_best_checkpoint": full_bank_linear_best_model_path,
            "full_bank_linear_best_snapshot_checkpoint": full_bank_linear_best_snapshot_path,
            "full_bank_linear_last_checkpoint": full_bank_linear_last_model_path,
            "full_bank_linear_best_per_class_csv": str(full_bank_linear_best_per_class_csv),
            "full_bank_linear_tsne_png": full_bank_linear_tsne_png,
            "full_bank_linear_prototype_report_csv": str(full_bank_linear_prototype_report_csv),
        },
    }
    if extra_summary_payload:
        summary_payload.update(extra_summary_payload)
    save_json(summary_json_path, summary_payload)

    return {
        "traditional_best_metrics": traditional_best_metrics,
        "single_label_best_metrics": single_label_best_metrics,
        "full_bank_linear_best_metrics": full_bank_linear_best_metrics,
    }


def _build_traditional_from_checkpoint(base_module, checkpoint_path: Path, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device)
    prototypes = state["prototypes"].detach().cpu()
    prototype_labels = state["prototype_labels"].detach().cpu().long()
    num_classes = int(state["last_layer.weight"].shape[0])
    model = base_module.TraditionalProtoPNetComparator(prototypes, prototype_labels, num_classes).to(device)
    model.load_state_dict(state)
    return model


def _build_single_label_from_checkpoint(base_module, checkpoint_path: Path, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device)
    prototypes = state["prototypes"].detach().cpu()
    prototype_label_matrix = state["prototype_label_matrix"].detach().cpu()
    num_classes = int(prototype_label_matrix.shape[1])
    model = base_module.ConfidenceWeightedProtoPNet(prototypes, prototype_label_matrix, num_classes).to(device)
    model.load_state_dict(state)
    return model


def _build_full_bank_linear_from_checkpoint(base_module, checkpoint_path: Path, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device)
    base_state = {key[len("base_model."):]: value for key, value in state.items() if key.startswith("base_model.")}
    if not base_state:
        raise ValueError(f"Full-bank-linear checkpoint is missing base_model keys: {checkpoint_path}")
    prototypes = base_state["prototypes"].detach().cpu()
    prototype_label_matrix = base_state["prototype_label_matrix"].detach().cpu()
    num_classes = int(prototype_label_matrix.shape[1])
    base_model = base_module.ConfidenceWeightedProtoPNet(prototypes, prototype_label_matrix, num_classes).to(device)
    model = base_module.ConfidenceWeightedWithLinearHead(base_model).to(device)
    model.load_state_dict(state)
    return model


def run_three_branch_eval_only_ci(
    *,
    base_module,
    task_name: str,
    output_root: Path,
    checkpoint_root: Path,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    bootstrap: int,
    seed: int,
    main_test_name: str = "test",
    primary_eval_split_name: str | None = None,
    extra_test_sets: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    test_loader = base_module.make_dataloader(x_test, y_test, base_module.EVAL_BATCH_SIZE, shuffle=False)
    eval_loaders: dict[str, tuple[Any, torch.Tensor]] = {main_test_name: (test_loader, y_test)}
    if extra_test_sets:
        for split_name, (x_extra, y_extra) in extra_test_sets.items():
            eval_loaders[split_name] = (
                base_module.make_dataloader(x_extra, y_extra, base_module.EVAL_BATCH_SIZE, shuffle=False),
                y_extra,
            )

    branch_specs = {
        "traditional_best": {
            "checkpoint": checkpoint_root / f"{task_name}_traditional_best.pt",
            "builder": _build_traditional_from_checkpoint,
        },
        "single_label_best": {
            "checkpoint": checkpoint_root / f"{task_name}_single_label_best.pt",
            "builder": _build_single_label_from_checkpoint,
        },
        "full_bank_linear_best": {
            "checkpoint": checkpoint_root / f"{task_name}_full_bank_linear_best.pt",
            "builder": _build_full_bank_linear_from_checkpoint,
        },
    }

    primary_split = primary_eval_split_name or main_test_name
    ci_output_dir = output_root / "ci"
    ci_output_dir.mkdir(parents=True, exist_ok=True)
    generated_reports: dict[str, Any] = {}

    for branch_name, spec in branch_specs.items():
        checkpoint_path = Path(spec["checkpoint"])
        if not checkpoint_path.exists():
            generated_reports[branch_name] = {
                "status": "missing_checkpoint",
                "checkpoint": str(checkpoint_path),
            }
            continue

        model = spec["builder"](base_module, checkpoint_path, device)
        branch_reports: dict[str, Any] = {"checkpoint": str(checkpoint_path), "splits": {}}

        for split_name, (loader, targets) in eval_loaders.items():
            logits = base_module.collect_logits(model, loader, device)
            y_true_np = targets.detach().cpu().numpy()
            y_score_np = torch.sigmoid(logits).detach().cpu().numpy()
            report_payload = build_multilabel_ci_report(
                class_names=list(base_module.TARGET_LABELS),
                y_true=y_true_np,
                y_score=y_score_np,
                bootstrap=int(bootstrap),
                seed=int(seed),
            )
            report_payload["metric_key"] = branch_name
            report_payload["eval_split_name"] = split_name
            metric_key = branch_name if split_name == primary_split else f"{branch_name}_{split_name}"
            refs = write_multilabel_ci_outputs(
                output_dir=ci_output_dir,
                metric_key=metric_key,
                report_payload=report_payload,
            )
            branch_reports["splits"][split_name] = refs

        generated_reports[branch_name] = branch_reports

    summary_payload = {
        "task_name": task_name,
        "bootstrap_iterations": int(bootstrap),
        "seed": int(seed),
        "primary_eval_split_name": primary_split,
        "reports": generated_reports,
    }
    save_ci_json(ci_output_dir / "aggregate_summary_ci.json", summary_payload)
    return summary_payload


def build_binary_target_labels(base_module, positive_label: str = "Chagas disease") -> dict[str, Any]:
    base_module.TARGET_LABELS = [positive_label]
    base_module.TARGET_PROMPTS = [f"An ECG recording showing {positive_label}."]
    base_module.TARGET_ALIAS_SETS = [{base_module.normalize_label_text(positive_label)}]
    base_module.TARGET_KEYWORDS = [sorted(base_module.TARGET_ALIAS_SETS[0])]
    base_module.NORMAL_LABEL = "Non-Chagas"
    base_module.NORMAL_CLASS_IDX = -1
    return {
        "task_name": "code15_chagas_binary",
        "columns": [positive_label],
        "labels": [positive_label],
        "normal_idx": None,
        "manifest_rows": [
            {
                "class_idx": 0,
                "canonical_label": positive_label,
                "source_column": positive_label,
                "prompt": f"An ECG recording showing {positive_label}.",
                "group": "CODE15 Chagas Binary",
                "alias_count": 1,
                "aliases": [positive_label],
            }
        ],
    }


def evaluate_binary_scores(
    y_true: torch.Tensor,
    y_logits: torch.Tensor,
    title: str,
) -> dict[str, Any]:
    y_true_np = y_true.detach().cpu().view(-1).numpy()
    y_score_np = torch.sigmoid(y_logits).detach().cpu().view(-1).numpy()

    roc_auc = float("nan")
    average_precision = float("nan")
    best_f1 = 0.0
    best_threshold = 0.5

    if len(np.unique(y_true_np)) > 1:
        try:
            from sklearn.metrics import roc_auc_score

            roc_auc = float(roc_auc_score(y_true_np, y_score_np))
        except Exception:
            roc_auc = float("nan")
        try:
            average_precision = float(average_precision_score(y_true_np, y_score_np))
        except Exception:
            average_precision = float("nan")

    for threshold in np.linspace(0.05, 0.95, 19):
        predictions = (y_score_np >= threshold).astype(np.int64)
        try:
            threshold_f1 = float(f1_score(y_true_np, predictions, zero_division=0))
        except Exception:
            threshold_f1 = 0.0
        if threshold_f1 > best_f1:
            best_f1 = threshold_f1
            best_threshold = float(threshold)

    payload = {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "positive_count": int(np.sum(y_true_np)),
        "negative_count": int(len(y_true_np) - np.sum(y_true_np)),
    }
    print(f"\n=== {title} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def train_three_branch_binary_experiment(
    *,
    base_module,
    task_name: str,
    output_root: Path,
    checkpoint_root: Path,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    train_idx_original,
    device: torch.device,
    max_epochs: int,
    label_selection: dict[str, Any],
    split_protocol: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    prototype_counts: list[int],
    summary_json_path: Path,
    extra_summary_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = train_three_branch_multilabel_experiment(
        base_module=base_module,
        task_name=task_name,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        train_idx_original=train_idx_original,
        device=device,
        max_epochs=max_epochs,
        label_selection=label_selection,
        label_stats_rows=[],
        mapping_warnings=[],
        split_protocol=split_protocol,
        manifest_rows=manifest_rows,
        prototype_counts=prototype_counts,
        summary_json_path=summary_json_path,
        primary_per_class_csv_path=output_root / "binary_metrics_full_bank_linear_best.csv",
        extra_summary_payload=extra_summary_payload,
    )
    return result
