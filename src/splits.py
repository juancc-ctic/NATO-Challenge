"""Construcción y escritura de splits leave-one-iteration-out por `split_group`.

Existe porque las filas de una iteración comparten vuelo, sensores y tiempo;
por eso separa por iteración completa y evita fuga entre train y validación.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Config
from src.parsing import json_safe, ms_to_iso


def _group_meta(detections: pd.DataFrame, group: str) -> dict[str, Any]:
    part = detections[detections["split_group"] == group]
    if part.empty:
        return {}
    first = part.iloc[0]
    return {
        "split": str(first["split"]),
        "dataset": str(first["dataset"]),
        "scenario": int(first["scenario"]),
        "iteration": int(first["iteration"]),
        "n_detections": int(len(part)),
        "n_sensors": int(part["sensor_id"].nunique()),
        "n_files": int(part["source_file"].nunique()),
        "t_min_iso": ms_to_iso(int(part["ts_ms"].min())),
        "t_max_iso": ms_to_iso(int(part["ts_ms"].max())),
    }


def build_splits(
    detections: pd.DataFrame,
    ground_truth: pd.DataFrame,
    friendly: pd.DataFrame,
    cfg: Config,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Construye folds LOO por iteración y comprueba que no haya fuga."""
    groups = sorted(str(value) for value in detections["split_group"].unique())
    dev_groups = [group for group in groups if group.startswith("test:")]
    holdout_groups = [group for group in groups if group.startswith("eval:")]
    gt_groups = set(str(value) for value in ground_truth["split_group"].unique())
    friendly_groups = set(str(value) for value in friendly["split_group"].unique())
    folds: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    if bool(cfg.splits.get("leave_one_iteration_out", True)):
        for index, group in enumerate(dev_groups):
            train = [item for item in dev_groups if item != group]
            folds.append({"fold": index, "train": train, "val": [group]})
            for item in train:
                fold_rows.append({"fold": index, "role": "train", "split_group": item})
            fold_rows.append({"fold": index, "role": "val", "split_group": group})
    else:
        folds.append({"fold": 0, "train": dev_groups, "val": []})
        for item in dev_groups:
            fold_rows.append({"fold": 0, "role": "train", "split_group": item})
    for fold in folds:
        overlap = set(fold["train"]) & set(fold["val"])
        if overlap:
            raise ValueError(f"Split leakage detected in fold {fold['fold']}: {sorted(overlap)}")
    manifest_rows: list[dict[str, Any]] = []
    for group in groups:
        meta = _group_meta(detections, group)
        if not meta:
            continue
        meta["split_group"] = group
        meta["role"] = "dev" if group in dev_groups else "holdout"
        meta["has_ground_truth"] = group in gt_groups
        meta["has_friendly_list"] = group in friendly_groups
        meta["n_ground_truth"] = int((ground_truth["split_group"] == group).sum()) if not ground_truth.empty else 0
        meta["n_friendly_serials"] = int(friendly[friendly["split_group"] == group]["serial_norm"].nunique()) if not friendly.empty else 0
        manifest_rows.append(meta)
    manifest = pd.DataFrame(manifest_rows)
    split_document = {
        "policy": str(cfg.splits.get("policy", "group_by_iteration")),
        "dev_split": str(cfg.splits.get("dev_split", "test")),
        "holdout_split": str(cfg.splits.get("holdout_split", "eval")),
        "leave_one_iteration_out": bool(cfg.splits.get("leave_one_iteration_out", True)),
        "random_state": int(cfg.splits.get("random_state", 42)),
        "dev_groups": dev_groups,
        "holdout_groups": holdout_groups,
        "folds": folds,
        "final": {"train": dev_groups, "predict": holdout_groups},
        "notes": [
            "Split by split_group, never by row.",
            "Supervised fitting, calibration and threshold selection use only dev groups with ground truth.",
            "Holdout groups are inference-only and never used for fitting or calibration.",
            "Fusion-node detections are test-only and must be excluded or flagged for eval inference.",
        ],
    }
    return split_document, manifest, pd.DataFrame(fold_rows)


def write_splits(
    split_document: dict[str, Any],
    manifest: pd.DataFrame,
    fold_rows: pd.DataFrame,
    cfg: Config,
) -> None:
    cfg.ensure_dirs()
    with open(cfg.interim / "splits.json", "w", encoding="utf-8") as handle:
        json.dump(json_safe(split_document), handle, indent=2, ensure_ascii=False)
    manifest.to_csv(cfg.reports / "split_manifest.csv", index=False)
    fold_rows.to_csv(cfg.reports / "split_folds.csv", index=False)


def run_splits(
    detections: pd.DataFrame,
    ground_truth: pd.DataFrame,
    friendly: pd.DataFrame,
    cfg: Config,
) -> dict[str, Any]:
    """Calcula y escribe `splits.json`, `split_manifest.csv` y `split_folds.csv`."""
    split_document, manifest, fold_rows = build_splits(
        detections, ground_truth, friendly, cfg
    )
    write_splits(split_document, manifest, fold_rows, cfg)
    return split_document
