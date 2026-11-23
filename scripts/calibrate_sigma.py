"""Calibra los sigma de medida con ground truth de `test`.

Fuentes:
- `reports/labels/sensors.csv` (etiquetado de Cayetana): error mediano por sensor
  e iteración para detecciones emparejadas. Se usa para sigma horizontal y azimut.
- Emparejamiento propio Húngaro ≤500 m: se usa para sigma vertical (`u`), que el
  etiquetado no separa.

No se ajusta con GOSPA. Salida: `reports/sigma_calibration.csv`.
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
# Mediana de |error| para una Gaussiana 1D: 0.6745·σ.
MEDIAN_ABS_TO_SIGMA = 0.6745
# Mediana de la distancia horizontal (Rayleigh 2D): 1.1774·σ.
RAYLEIGH_MEDIAN_TO_SIGMA = 1.1774


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
        rows_by_time = {
            int(ts): frame for ts, frame in group.groupby("ts_ms", sort=False)
        }
        positions_by_time = {
            ts: frame[["n", "e", "u"]].to_numpy(dtype=float)
            for ts, frame in rows_by_time.items()
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
            matched = rows_by_time[int(timestamp)].iloc[track_index].copy()
            errors = detections_at_ts[track_index] - gt_positions[gt_index]
            matched["dn"] = errors[:, 0]
            matched["de"] = errors[:, 1]
            matched["du"] = errors[:, 2]
            output.append(matched)
    if not output:
        return pd.DataFrame()
    return pd.concat(output, ignore_index=True)


def _matched_sigma(matched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sensor_type, group in matched.groupby("sensor_type", sort=False):
        row: dict[str, object] = {"sensor_type": sensor_type}
        for axis in ("n", "e", "u"):
            errors = group[f"d{axis}"].to_numpy(dtype=float)
            errors = errors[np.isfinite(errors)]
            if len(errors) == 0:
                continue
            median = float(np.median(errors))
            row[f"mad_sigma_{axis}_m"] = float(
                MAD_TO_SIGMA * np.median(np.abs(errors - median))
            )
            row[f"rms_{axis}_m"] = float(np.sqrt(np.mean(errors**2)))
        rows.append(row)
    return pd.DataFrame(rows)


def _labels_sigma(sensors_path: Path) -> pd.DataFrame:
    if not sensors_path.exists():
        return pd.DataFrame()
    sensors = pd.read_csv(sensors_path).dropna(subset=["mode"])
    bearing_types = {"rf", "acoustic", "camera"}
    rows: list[dict[str, object]] = []
    for sensor_type in sorted(sensors["sensor_type"].unique()):
        preferred = "brg" if sensor_type in bearing_types else "pos"
        group = sensors[
            (sensors["sensor_type"] == sensor_type)
            & (sensors["mode"] == preferred)
        ]
        if group.empty:
            group = sensors[sensors["sensor_type"] == sensor_type]
            preferred = str(group["mode"].iloc[0])
        median = float(np.median(group["err_median"].dropna()))
        if preferred == "pos":
            rows.append(
                {
                    "sensor_type": sensor_type,
                    "mode": preferred,
                    "sigma_h_m": median / RAYLEIGH_MEDIAN_TO_SIGMA,
                    "sigma_az_deg": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "sensor_type": sensor_type,
                    "mode": preferred,
                    "sigma_h_m": np.nan,
                    "sigma_az_deg": median / MEDIAN_ABS_TO_SIGMA,
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
    matched_sigma = _matched_sigma(matched)
    labels_sigma = _labels_sigma(cfg.reports / "labels" / "sensors.csv")

    rows: list[dict[str, object]] = []
    sensor_types = set(matched_sigma.get("sensor_type", [])) | set(
        labels_sigma.get("sensor_type", [])
    )
    for sensor_type in sorted(sensor_types):
        m = matched_sigma[matched_sigma["sensor_type"] == sensor_type]
        l = labels_sigma[labels_sigma["sensor_type"] == sensor_type]
        mode = "brg" if sensor_type in {"rf", "acoustic", "camera"} else "pos"
        row: dict[str, object] = {"sensor_type": sensor_type, "mode": mode}
        if mode == "pos":
            if len(l) and np.isfinite(l.iloc[0]["sigma_h_m"]):
                row["sigma_n_m"] = float(l.iloc[0]["sigma_h_m"])
                row["sigma_e_m"] = float(l.iloc[0]["sigma_h_m"])
                row["basis_h"] = "labels_median"
            elif len(m):
                row["sigma_n_m"] = float(m.iloc[0]["mad_sigma_n_m"])
                row["sigma_e_m"] = float(m.iloc[0]["mad_sigma_e_m"])
                row["basis_h"] = "hungarian_mad"
            row["sigma_u_m"] = (
                float(m.iloc[0]["mad_sigma_u_m"]) if len(m) else np.nan
            )
            row["sigma_az_deg"] = np.nan
        else:
            row["sigma_n_m"] = np.nan
            row["sigma_e_m"] = np.nan
            row["sigma_u_m"] = np.nan
            row["sigma_az_deg"] = (
                float(l.iloc[0]["sigma_az_deg"]) if len(l) else np.nan
            )
            row["basis_h"] = "labels_median" if len(l) else "none"
        rows.append(row)

    summary = pd.DataFrame(rows).round(3)
    output = cfg.reports / "sigma_calibration.csv"
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"\nEscrito: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
