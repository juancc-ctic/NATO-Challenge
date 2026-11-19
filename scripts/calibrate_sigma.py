"""Calibra el sigma de posición con ground truth de `test`.

Empareja detecciones de posición con GT por timestamp (Húngaro, ≤500 m) y
calcula, por tipo de sensor y eje: mediana del error, sigma robusto (MAD) y RMS.
El sigma robusto es la base de `etl.default_sigma_m`; no se ajusta con GOSPA.

Uso: `python scripts/calibrate_sigma.py`
Salida: `reports/sigma_calibration.csv`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config

MAX_MATCH_M = 500.0
MAD_TO_SIGMA = 1.4826


def _interpolate_gt(
    ground_truth: pd.DataFrame,
    timestamps: np.ndarray,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for _, group in ground_truth.groupby("drone_key", sort=False):
        ordered = group.sort_values("ts_ms")
        times = ordered["ts_ms"].to_numpy(dtype=np.int64)
        mask = (timestamps >= times[0]) & (timestamps <= times[-1])
        if not mask.any():
            continue
        values = np.full((len(timestamps), 3), np.nan, dtype=float)
        for axis, column in enumerate(("n", "e", "u")):
            values[mask, axis] = np.interp(
                timestamps[mask],
                times,
                ordered[column].to_numpy(dtype=float),
            )
        rows.append(values)
    if not rows:
        return np.empty((len(timestamps), 0, 3), dtype=float)
    return np.stack(rows, axis=1)


def _matched_errors(
    detections: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for split_group, group in detections.groupby("split_group", sort=False):
        timestamps = np.sort(group["ts_ms"].unique())
        gt_matrix = _interpolate_gt(
            ground_truth[ground_truth["split_group"] == split_group],
            timestamps,
        )
        positions_by_time = {
            int(ts): frame[["n", "e", "u"]].to_numpy(dtype=float)
            for ts, frame in group.groupby("ts_ms", sort=False)
        }
        rows_by_time = {
            int(ts): frame
            for ts, frame in group.groupby("ts_ms", sort=False)
        }
        for index, timestamp in enumerate(timestamps):
            gt_positions = gt_matrix[index]
            gt_positions = gt_positions[~np.isnan(gt_positions).any(axis=1)]
            if len(gt_positions) == 0:
                continue
            detections_at_ts = positions_by_time.get(int(timestamp))
            if detections_at_ts is None or len(detections_at_ts) == 0:
                continue
            distances = np.linalg.norm(
                detections_at_ts[:, None, :] - gt_positions[None, :, :],
                axis=2,
            )
            cost = distances.copy()
            cost[distances > MAX_MATCH_M] = 1e9
            track_index, gt_index = linear_sum_assignment(cost)
            keep = distances[track_index, gt_index] <= MAX_MATCH_M
            track_index = track_index[keep]
            gt_index = gt_index[keep]
            if len(track_index) == 0:
                continue
            frame = rows_by_time[int(timestamp)]
            matched = frame.iloc[track_index].copy()
            errors = detections_at_ts[track_index] - gt_positions[gt_index]
            matched["dn"] = errors[:, 0]
            matched["de"] = errors[:, 1]
            matched["du"] = errors[:, 2]
            output.append(matched)
    if not output:
        return pd.DataFrame()
    return pd.concat(output, ignore_index=True)


def _summarise(matched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sensor_type, group in matched.groupby("sensor_type", sort=False):
        for axis in ("n", "e", "u"):
            errors = group[f"d{axis}"].to_numpy(dtype=float)
            errors = errors[np.isfinite(errors)]
            if len(errors) == 0:
                continue
            median = float(np.median(errors))
            mad_sigma = float(
                MAD_TO_SIGMA * np.median(np.abs(errors - median))
            )
            rms = float(np.sqrt(np.mean(errors**2)))
            reported_rows = group[
                group["sigma_source"].isin(
                    ["reported_m", "reported_deg_converted"]
                )
            ]
            reported = reported_rows[f"sigma_{axis}"].to_numpy(dtype=float)
            reported = reported[np.isfinite(reported)]
            reported_errors = reported_rows[f"d{axis}"].to_numpy(dtype=float)
            reported_errors = reported_errors[np.isfinite(reported_errors)]
            rows.append(
                {
                    "sensor_type": sensor_type,
                    "axis": axis,
                    "n_matched": int(len(errors)),
                    "median_error_m": median,
                    "mad_sigma_m": mad_sigma,
                    "rms_m": rms,
                    "n_reported": int(len(reported)),
                    "reported_median_m": (
                        float(np.median(reported)) if len(reported) else np.nan
                    ),
                    "nis_reported": (
                        float(
                            np.mean(reported_errors**2)
                            / np.median(reported) ** 2
                        )
                        if len(reported) and len(reported_errors)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    detections = pd.read_parquet(
        cfg.interim / "detections.parquet",
        columns=[
            "split",
            "split_group",
            "ts_ms",
            "sensor_type",
            "has_position",
            "n",
            "e",
            "u",
            "sigma_n",
            "sigma_e",
            "sigma_u",
            "sigma_source",
        ],
    )
    ground_truth = pd.read_parquet(cfg.interim / "ground_truth.parquet")
    detections = detections[
        (detections["split"] == "test")
        & detections["has_position"].fillna(False).astype(bool)
    ]
    ground_truth = ground_truth[
        ground_truth["split_group"].isin(detections["split_group"].unique())
    ]
    matched = _matched_errors(detections, ground_truth)
    summary = _summarise(matched)
    output = cfg.reports / "sigma_calibration.csv"
    summary.to_csv(output, index=False)
    print(summary.round(3).to_string(index=False))
    print(f"\nEscrito: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
