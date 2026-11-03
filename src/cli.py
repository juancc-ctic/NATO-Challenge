"""CLI del pipeline: `inventory`, `etl`, `splits`, `check` y `all`.

Existe para ejecutar el pipeline por pasos y de forma reproducible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.check import run_check
from src.config import load_config
from src.etl import run_etl
from src.parsing import json_safe
from src.splits import run_splits


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(json_safe(payload), indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de `python -m src.cli`."""
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    parser.add_argument("command", choices=["check", "inventory", "etl", "splits", "all"])
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

    result = run_etl(cfg, write_interim=True)
    document = run_splits(
        result.detections, result.ground_truth, result.friendly, cfg
    )
    check_document = run_check(cfg)
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
