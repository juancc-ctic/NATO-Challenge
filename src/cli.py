"""CLI del pipeline: `inventory`, `etl`, `splits`, `track`, `evaluate`, `check` y `all`.

Existe para ejecutar el pipeline por pasos y de forma reproducible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.check import run_check
from src.config import load_config
from src.etl import run_etl
from src.evaluate import run_evaluation
from src.parsing import json_safe
from src.splits import run_splits
from src.tracker import run_tracker


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(json_safe(payload), indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de `python -m src.cli`."""
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    parser.add_argument(
        "command",
        choices=["check", "inventory", "etl", "splits", "track", "evaluate", "all"],
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    cfg.ensure_dirs()

    if args.command == "check":
        _print(run_check(cfg))
        return 0

    if args.command == "inventory":
        result = run_etl(cfg, write_interim=False)
        _print(
            {
                "counts": result.summary["counts"],
                "by_sensor_type": result.summary["by_sensor_type"],
                "by_split": result.summary["by_split"],
                "reports": {
                    "inventory": str(cfg.reports / "inventory.csv"),
                    "issues": str(cfg.reports / "etl_issues.csv"),
                    "summary": str(cfg.reports / "etl_summary.json"),
                },
            }
        )
        return 0

    if args.command == "etl":
        result = run_etl(cfg, write_interim=True)
        _print(
            {
                "counts": result.summary["counts"],
                "by_sensor_type": result.summary["by_sensor_type"],
                "by_split": result.summary["by_split"],
                "tables": result.summary["tables"],
                "interim": str(cfg.interim),
                "reports": str(cfg.reports),
            }
        )
        return 0

    if args.command == "splits":
        result = run_etl(cfg, write_interim=False)
        document = run_splits(
            result.detections, result.ground_truth, result.friendly, cfg
        )
        _print(
            {
                "policy": document["policy"],
                "dev_groups": document["dev_groups"],
                "holdout_groups": document["holdout_groups"],
                "n_folds": len(document["folds"]),
                "reports": {
                    "splits": str(cfg.interim / "splits.json"),
                    "manifest": str(cfg.reports / "split_manifest.csv"),
                    "folds": str(cfg.reports / "split_folds.csv"),
                },
            }
        )
        return 0

    if args.command == "track":
        detections = pd.read_parquet(cfg.interim / "detections.parquet")
        tracks = run_tracker(detections, cfg)
        output = cfg.reports / "tracks.parquet"
        tracks.to_parquet(output, index=False)
        _print(
            {
                "tracks": int(len(tracks)),
                "track_ids": int(tracks["track_id"].nunique()),
                "split_groups": int(tracks["split_group"].nunique()),
                "output": str(output),
            }
        )
        return 0

    if args.command == "evaluate":
        tracks = pd.read_parquet(cfg.reports / "tracks.parquet")
        ground_truth = pd.read_parquet(cfg.interim / "ground_truth.parquet")
        result = run_evaluation(tracks, ground_truth, cfg)
        _print(result)
        return 0

    result = run_etl(cfg, write_interim=True)
    document = run_splits(
        result.detections, result.ground_truth, result.friendly, cfg
    )
    check_document = run_check(cfg)
    tracks = run_tracker(result.detections, cfg)
    tracks.to_parquet(cfg.reports / "tracks.parquet", index=False)
    metrics = run_evaluation(tracks, result.ground_truth, cfg)
    _print(
        {
            "etl": {
                "counts": result.summary["counts"],
                "by_sensor_type": result.summary["by_sensor_type"],
                "by_split": result.summary["by_split"],
                "tables": result.summary["tables"],
            },
            "splits": {
                "policy": document["policy"],
                "dev_groups": document["dev_groups"],
                "holdout_groups": document["holdout_groups"],
                "n_folds": len(document["folds"]),
            },
            "check": {
                "n_templates": check_document["n_templates"],
                "n_ok": check_document["n_ok"],
            },
            "tracking": {
                "tracks": int(len(tracks)),
                "track_ids": int(tracks["track_id"].nunique()),
                "gospa": metrics["gospa"],
                "rmse": metrics["rmse"],
                "recall": metrics["recall"],
                "false_track_ratio": metrics["false_track_ratio"],
                "id_switches": metrics["id_switches"],
            },
            "outputs": {
                "interim": str(cfg.interim),
                "reports": str(cfg.reports),
                "out": str(cfg.out),
            },
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
