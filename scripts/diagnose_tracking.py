"""Diagnóstico reproducible del tracking.

Calcula, por iteración, la métrica principal GOSPA(c=50, p=2) y la secundaria
GOSPA(c=100, p=2), su descomposición (localización, GT no detectados y tracks
falsos), la cobertura GT->detección, la cobertura de solo-azimut, duplicados y
estadísticas de vida de los tracks.

No modifica el tracker ni `config.yaml`; solo lee `reports/tracks.parquet`,
`data/interim/detections.parquet` y `data/interim/ground_truth.parquet`, y
escribe `reports/tracking_diagnostics.csv` y `.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config


def _interpolate_gt(
    ground_truth: pd.DataFrame,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Interpola el GT de cada dron a los timestamps de track.

    Devuelve una matriz `(n_timestamps, n_drones, 3)` con NaN fuera de la vida
    de cada dron y la lista de `drone_key` en el mismo orden.
    """
    rows: list[np.ndarray] = []
    keys: list[str] = []
    for drone_key, group in ground_truth.groupby("drone_key", sort=False):
        ordered = group.sort_values("ts_ms")
        times = ordered["ts_ms"].to_numpy(dtype=np.int64)
        mask = (timestamps >= times[0]) & (timestamps <= times[-1])
        if not mask.any():
            continue
        values = np.full((len(timestamps), 3), np.nan, dtype=float)
        for axis, column in enumerate(("n", "e", "u")):
            values[mask, axis] = np.interp(
                timestamps[mask], times, ordered[column].to_numpy(dtype=float)
            )
        rows.append(values)
        keys.append(str(drone_key))
    if not rows:
        return np.empty((len(timestamps), 0, 3), dtype=float), []
    return np.stack(rows, axis=1), keys


def _evaluate_group(
    tracks: pd.DataFrame,
    ground_truth: pd.DataFrame,
    c: float,
    p: float,
    match_distance: float,
) -> dict[str, Any]:
    """Calcula GOSPA y su descomposición para una iteración."""
    timestamps = np.sort(tracks["ts_ms"].unique())
    gt_matrix, gt_keys = _interpolate_gt(ground_truth, timestamps)
    positions_by_time = {
        int(timestamp): frame[["n", "e", "u"]].to_numpy(dtype=float)
        for timestamp, frame in tracks.groupby("ts_ms", sort=False)
    }
    track_ids_by_time = {
        int(timestamp): frame["track_id"].astype(str).tolist()
        for timestamp, frame in tracks.groupby("ts_ms", sort=False)
    }
    gospa_values: list[float] = []
    squared_errors: list[float] = []
    horizontal_errors: list[float] = []
    vertical_errors: list[float] = []
    missed_per_ts: list[int] = []
    false_per_ts: list[int] = []
    gt_per_ts: list[int] = []
    tracks_per_ts: list[int] = []
    matched_total = 0
    gt_total = 0
    track_total = 0
    false_total = 0
    id_switches = 0
    last_track: dict[str, str] = {}
    for index, timestamp in enumerate(timestamps):
        gt_positions = gt_matrix[index]
        valid_gt = ~np.isnan(gt_positions).any(axis=1)
        gt_positions = gt_positions[valid_gt]
        current_gt_keys = [key for key, valid in zip(gt_keys, valid_gt) if valid]
        if len(gt_positions) == 0:
            continue
        track_positions = positions_by_time.get(int(timestamp), np.empty((0, 3)))
        track_ids = track_ids_by_time.get(int(timestamp), [])
        distances = np.linalg.norm(
            track_positions[:, None, :] - gt_positions[None, :, :], axis=2
        )
        cost = distances.copy()
        cost[distances > c] = c * 1.0e6
        rows, cols = linear_sum_assignment(cost)
        matched = [
            (int(row), int(col), float(distances[row, col]))
            for row, col in zip(rows, cols)
            if distances[row, col] <= c
        ]
        localization = sum(distance**p for _, _, distance in matched)
        missed = len(gt_positions) - len(matched)
        false = len(track_positions) - len(matched)
        cardinality = (c**p / p) * (missed + false)
        gospa_values.append((localization + cardinality) ** (1.0 / p))
        squared_errors.extend([distance**2 for _, _, distance in matched])
        horizontal_errors.extend(
            [
                float(np.linalg.norm(track_positions[row, :2] - gt_positions[col, :2])) ** 2
                for row, col, _ in matched
            ]
        )
        vertical_errors.extend(
            [
                float(track_positions[row, 2] - gt_positions[col, 2]) ** 2
                for row, col, _ in matched
            ]
        )
        missed_per_ts.append(missed)
        false_per_ts.append(false)
        gt_per_ts.append(len(gt_positions))
        tracks_per_ts.append(len(track_positions))
        matched_total += len(matched)
        gt_total += len(gt_positions)
        track_total += len(track_positions)
        false_total += false
        for gt_index, gt_key in enumerate(current_gt_keys):
            if len(track_positions) == 0:
                continue
            distances_to_gt = np.linalg.norm(
                track_positions - gt_positions[gt_index][None, :], axis=1
            )
            nearest = int(np.argmin(distances_to_gt))
            if distances_to_gt[nearest] > match_distance:
                continue
            previous = last_track.get(gt_key)
            if previous is not None and previous != track_ids[nearest]:
                id_switches += 1
            last_track[gt_key] = track_ids[nearest]
    if not gospa_values:
        return {
            "gospa": float("nan"),
            "rmse": float("nan"),
            "recall": 0.0,
            "false_ratio": 0.0,
            "id_switches": 0,
            "n_timestamps": 0,
            "gt_per_ts": float("nan"),
            "tracks_per_ts": float("nan"),
            "missed_per_ts": float("nan"),
            "false_per_ts": float("nan"),
            "h_rmse": float("nan"),
            "v_rmse": float("nan"),
        }
    return {
        "gospa": float(np.mean(gospa_values)),
        "rmse": float(np.sqrt(np.mean(squared_errors))) if squared_errors else float("nan"),
        "recall": float(matched_total / gt_total) if gt_total else 0.0,
        "false_ratio": float(false_total / track_total) if track_total else 0.0,
        "id_switches": int(id_switches),
        "n_timestamps": int(len(gospa_values)),
        "gt_per_ts": float(np.mean(gt_per_ts)),
        "tracks_per_ts": float(np.mean(tracks_per_ts)),
        "missed_per_ts": float(np.mean(missed_per_ts)),
        "false_per_ts": float(np.mean(false_per_ts)),
        "h_rmse": float(np.sqrt(np.mean(horizontal_errors))) if horizontal_errors else float("nan"),
        "v_rmse": float(np.sqrt(np.mean(vertical_errors))) if vertical_errors else float("nan"),
    }


def _coverage(
    detections: pd.DataFrame,
    ground_truth: pd.DataFrame,
    window_ms: int = 750,
) -> dict[str, float]:
    """Cobertura GT->detección de posición y precisión de detección."""
    position = detections[
        detections["has_position"].fillna(False).astype(bool)
        & detections["n"].notna()
        & detections["e"].notna()
    ].sort_values("ts_ms")
    if position.empty or ground_truth.empty:
        return {
            "gt_det_le50": float("nan"),
            "gt_det_le100": float("nan"),
            "det_precision_le50": float("nan"),
        }
    det_times = position["ts_ms"].to_numpy(dtype=np.int64)
    det_values = position[["n", "e", "u"]].to_numpy(dtype=float)
    gt_times = ground_truth["ts_ms"].to_numpy(dtype=np.int64)
    gt_values = ground_truth[["n", "e", "u"]].to_numpy(dtype=float)
    nearest_for_gt: list[float] = []
    for timestamp, value in zip(gt_times, gt_values):
        start = np.searchsorted(det_times, timestamp - window_ms)
        end = np.searchsorted(det_times, timestamp + window_ms)
        if end <= start:
            nearest_for_gt.append(float("inf"))
            continue
        distances = np.linalg.norm(det_values[start:end] - value[None, :], axis=1)
        nearest_for_gt.append(float(np.nanmin(distances)))
    nearest_for_gt = np.asarray(nearest_for_gt, dtype=float)
    nearest_for_det: list[float] = []
    for timestamp, value in zip(det_times, det_values):
        start = np.searchsorted(gt_times, timestamp - window_ms)
        end = np.searchsorted(gt_times, timestamp + window_ms)
        if end <= start:
            nearest_for_det.append(float("inf"))
            continue
        distances = np.linalg.norm(gt_values[start:end] - value[None, :], axis=1)
        nearest_for_det.append(float(np.nanmin(distances)))
    nearest_for_det = np.asarray(nearest_for_det, dtype=float)
    return {
        "gt_det_le50": float(np.mean(nearest_for_gt <= 50.0)),
        "gt_det_le100": float(np.mean(nearest_for_gt <= 100.0)),
        "det_precision_le50": float(np.mean(nearest_for_det <= 50.0)),
    }


def _bearing_coverage(
    detections: pd.DataFrame,
    ground_truth: pd.DataFrame,
    window_ms: int = 750,
    tolerance_deg: float = 5.0,
) -> dict[str, float]:
    """Cobertura GT->bearing y cuántos sensores ven cada GT a la vez."""
    bearing = detections[
        detections["has_bearing"].fillna(False).astype(bool)
        & detections["az_deg"].notna()
        & detections["has_sensor_location"].fillna(False).astype(bool)
    ].sort_values("ts_ms")
    if bearing.empty or ground_truth.empty:
        return {"gt_bearing_le5": float("nan"), "gt_bearing2_le5": float("nan")}
    times = bearing["ts_ms"].to_numpy(dtype=np.int64)
    sensor_n = bearing["sensor_n"].to_numpy(dtype=float)
    sensor_e = bearing["sensor_e"].to_numpy(dtype=float)
    azimuth = bearing["az_deg"].to_numpy(dtype=float)
    sensor_id = bearing["sensor_id"].astype(str).to_numpy()
    one_sensor: list[int] = []
    two_sensors: list[int] = []
    for _, row in ground_truth.iterrows():
        start = np.searchsorted(times, row["ts_ms"] - window_ms)
        end = np.searchsorted(times, row["ts_ms"] + window_ms)
        if end <= start:
            one_sensor.append(0)
            two_sensors.append(0)
            continue
        dn = sensor_n[start:end] - float(row["n"])
        de = sensor_e[start:end] - float(row["e"])
        expected = np.degrees(np.arctan2(de, dn)) % 360.0
        difference = np.abs((expected - azimuth[start:end] + 180.0) % 360.0 - 180.0)
        visible = set(sensor_id[start:end][difference <= tolerance_deg])
        one_sensor.append(1 if len(visible) >= 1 else 0)
        two_sensors.append(1 if len(visible) >= 2 else 0)
    return {
        "gt_bearing_le5": float(np.mean(one_sensor)),
        "gt_bearing2_le5": float(np.mean(two_sensors)),
    }


def _track_stats(tracks: pd.DataFrame) -> dict[str, Any]:
    """Estadísticas de vida y estados de los tracks."""
    lifespan = tracks.groupby(["split_group", "track_id"]).size()
    last_status = tracks.groupby(["split_group", "track_id"])["status"].last()
    status = tracks["status"].value_counts(normalize=True)
    return {
        "n_track_ids": int(lifespan.size),
        "n_track_rows": int(len(tracks)),
        "median_lifespan": float(lifespan.median()) if len(lifespan) else float("nan"),
        "coast_ratio": float(status.get("coast", 0.0)),
        "tentative_ratio": float(status.get("tentative", 0.0)),
        "confirmed_ratio": float(status.get("confirmed", 0.0)),
        "last_status_confirmed": int((last_status == "confirmed").sum()),
    }


def _duplicates(detections: pd.DataFrame) -> dict[str, int]:
    """Duplicados por clave natural dentro de la misma iteración y sensor."""
    position = detections[detections["has_position"].fillna(False).astype(bool)]
    key = ["split_group", "sensor_id", "ts_ms", "n", "e", "u"]
    extra = int(position.duplicated(subset=key).sum())
    near = position[np.hypot(position["n"], position["e"]) <= 5000.0]
    extra_near = int(near.duplicated(subset=key).sum())
    return {"duplicate_extra": extra, "duplicate_extra_near": extra_near}


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    tracks = pd.read_parquet(cfg.reports / "tracks.parquet")
    detections = pd.read_parquet(cfg.interim / "detections.parquet")
    ground_truth = pd.read_parquet(cfg.interim / "ground_truth.parquet")
    evaluation = cfg.data.get("evaluation", {})
    c_primary = float(evaluation.get("gospa_c_m", 50.0))
    c_reference = float(evaluation.get("gospa_c_reference_m", 100.0))
    p = float(evaluation.get("gospa_p", 2.0))
    match_distance = float(evaluation.get("match_distance_m", 50.0))
    rows: list[dict[str, Any]] = []
    for split_group, track_group in tracks.groupby("split_group", sort=True):
        gt_group = ground_truth[ground_truth["split_group"] == split_group]
        row: dict[str, Any] = {"split_group": str(split_group), "has_gt": bool(len(gt_group))}
        row.update(_track_stats(track_group))
        row.update(_duplicates(detections[detections["split_group"] == split_group]))
        if len(gt_group):
            primary = _evaluate_group(track_group, gt_group, c_primary, p, match_distance)
            reference = _evaluate_group(track_group, gt_group, c_reference, p, match_distance)
            row.update({f"gospa_c{int(c_primary)}": primary["gospa"]})
            row.update({f"gospa_c{int(c_reference)}": reference["gospa"]})
            row.update({f"rmse_c{int(c_primary)}": primary["rmse"]})
            row.update({f"recall_c{int(c_primary)}": primary["recall"]})
            row.update({f"false_ratio_c{int(c_primary)}": primary["false_ratio"]})
            row.update({f"id_switches_c{int(c_primary)}": primary["id_switches"]})
            row.update({f"n_timestamps_c{int(c_primary)}": primary["n_timestamps"]})
            row.update(
                {
                    "gt_per_ts": primary["gt_per_ts"],
                    "tracks_per_ts": primary["tracks_per_ts"],
                    "missed_per_ts": primary["missed_per_ts"],
                    "false_per_ts": primary["false_per_ts"],
                    "h_rmse": primary["h_rmse"],
                    "v_rmse": primary["v_rmse"],
                }
            )
            row.update(_coverage(detections[detections["split_group"] == split_group], gt_group))
            row.update(_bearing_coverage(detections[detections["split_group"] == split_group], gt_group))
        rows.append(row)
    diagnostics = pd.DataFrame(rows)
    output_csv = cfg.reports / "tracking_diagnostics.csv"
    diagnostics.to_csv(output_csv, index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256_16": _config_hash(config_path),
        "gospa_c_primary_m": c_primary,
        "gospa_c_reference_m": c_reference,
        "gospa_p": p,
        "tracks_parquet": str(cfg.reports / "tracks.parquet"),
        "output_csv": str(output_csv),
        "overall": {
            "gospa_c_primary": float(diagnostics[f"gospa_c{int(c_primary)}"].mean()),
            "gospa_c_reference": float(diagnostics[f"gospa_c{int(c_reference)}"].mean()),
            "recall_c_primary": float(diagnostics[f"recall_c{int(c_primary)}"].mean()),
            "false_ratio_c_primary": float(diagnostics[f"false_ratio_c{int(c_primary)}"].mean()),
            "gt_det_le50": float(diagnostics["gt_det_le50"].mean()),
            "gt_det_le100": float(diagnostics["gt_det_le100"].mean()),
            "gt_bearing_le5": float(diagnostics["gt_bearing_le5"].mean()),
            "gt_bearing2_le5": float(diagnostics["gt_bearing2_le5"].mean()),
            "duplicate_extra": int(diagnostics["duplicate_extra"].sum()),
            "duplicate_extra_near": int(diagnostics["duplicate_extra_near"].sum()),
        },
    }
    output_json = cfg.reports / "tracking_diagnostics.json"
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    summary = run(Path(args.config).resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
