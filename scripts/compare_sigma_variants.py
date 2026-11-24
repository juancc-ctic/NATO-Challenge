"""Compara la calibración de sigma MAD frente a la de labels.

Variantes:
- `labels`: valores actuales de `config.yaml` (mediana de
  `reports/labels/sensors.csv`).
- `mad`: valores de `479bfab` (Húngaro + 1.4826·MAD).

Usa el mismo tracker y la misma evaluación; solo cambian los sigma. Por defecto
evalúa solo `test`, que es el único split con ground truth.

Uso: `python scripts/compare_sigma_variants.py`
Salida: `reports/sigma_mad_vs_labels/`
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.tracker import run_tracker  # noqa: E402

MAD_SIGMA_M = {
    "radar": {"n": 150.0, "e": 154.0, "u": 65.0},
    "cyber": {"n": 41.0, "e": 38.0, "u": 30.0},
    "fusion": {"n": 29.0, "e": 31.0, "u": 18.0},
    "other": {"n": 5.0, "e": 7.0, "u": 2.0},
}
MAD_SIGMA_AZ_DEG = {"rf": 5.0, "acoustic": 10.0, "camera": 1.0}


def _variant(cfg, name: str):
    data = copy.deepcopy(cfg.data)
    if name == "mad":
        data["etl"]["default_sigma_m"] = copy.deepcopy(MAD_SIGMA_M)
        data["tracker"]["measurement"]["sigma_az_deg"] = copy.deepcopy(
            MAD_SIGMA_AZ_DEG
        )
    return replace(cfg, data=data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("test", "all"),
        default="test",
        help="test (por defecto) o all",
    )
    parser.add_argument(
        "--no-save-tracks",
        action="store_true",
        help="no guarda los tracks de cada variante",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / "config.yaml")
    detections = pd.read_parquet(cfg.interim / "detections.parquet")
    ground_truth = pd.read_parquet(cfg.interim / "ground_truth.parquet")
    if args.split == "test":
        detections = detections[detections["split"] == "test"]
    ground_truth = ground_truth[
        ground_truth["split_group"].isin(detections["split_group"].unique())
    ]

    outdir = cfg.reports / "sigma_mad_vs_labels"
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for name in ("labels", "mad"):
        variant_cfg = _variant(cfg, name)
        tracks = run_tracker(detections, variant_cfg)
        result = evaluate(tracks, ground_truth, variant_cfg)
        if not args.no_save_tracks:
            tracks.to_parquet(outdir / f"{name}_tracks.parquet", index=False)
        for group in result["groups"]:
            rows.append({"variant": name, **group})
        print(
            f"{name}: GOSPA={result['gospa']:.3f} "
            f"recall={result['recall']:.3f} "
            f"false={result['false_track_ratio']:.3f} "
            f"id_switches={result['id_switches']} "
            f"tracks={len(tracks)}"
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "metrics.csv", index=False)
    summary = {
        name: {
            "gospa": float(part["gospa"].mean()),
            "recall": float(part["recall"].mean()),
            "false_track_ratio": float(part["false_track_ratio"].mean()),
            "id_switches": int(part["id_switches"].sum()),
        }
        for name, part in metrics.groupby("variant", sort=False)
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print(f"\nEscrito: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
