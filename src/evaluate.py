"""Evaluación de tracks contra ground truth.

Flujo: evaluación, después del tracker. Calcula GOSPA, RMSE, recall, tracks
falsos e ID switches por iteración; no interviene en predicción, gating,
asociación, actualización ni ciclo de vida.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _subcfg(cfg: Any, key: str) -> Mapping[str, Any]:
    if hasattr(cfg, "data"):
        return cfg.data.get(key, {}) or {}
    if isinstance(cfg, Mapping):
        value = cfg.get(key, {})
        return value if isinstance(value, Mapping) else {}
    return {}


def _positions(frame: pd.DataFrame) -> np.ndarray:
    """[evaluación] Extrae posiciones `n,e,u` como matriz.

    Pertenece a evaluación: prepara datos para las métricas.
    """
    values = frame[["n", "e", "u"]].to_numpy(dtype=float)
    return np.nan_to_num(values, nan=0.0)


def _match_positions(
    track_positions: np.ndarray,
    gt_positions: np.ndarray,
    c: float,
) -> list[tuple[int, int, float]]:
    """[evaluación] Empareja tracks y GT con coste truncado.

    Pertenece a evaluación: es la asociación interna de GOSPA.
    """
    if len(track_positions) == 0 or len(gt_positions) == 0:
        return []
    distances = np.linalg.norm(
        track_positions[:, None, :] - gt_positions[None, :, :],
        axis=2,
    )
    # Los pares a más de c no cuentan como emparejados.
    cost = distances.copy()
    cost[distances > c] = c * 1.0e6
    rows, cols = linear_sum_assignment(cost)
    return [
        (int(row), int(col), float(distances[row, col]))
        for row, col in zip(rows, cols)
        if distances[row, col] <= c
    ]


def gospa(
    track_positions: np.ndarray,
    gt_positions: np.ndarray,
    c: float = 100.0,
    p: float = 2.0,
) -> tuple[float, list[tuple[int, int, float]]]:
    """[evaluación] Calcula GOSPA y los pares emparejados.

    Pertenece a evaluación: es una métrica posterior al tracker.
    """
    # Localización de pares + penalización por GT no detectado y tracks falsos.
    matched = _match_positions(track_positions, gt_positions, c)
    matched_count = len(matched)
    localization = sum(distance**p for _, _, distance in matched)
    missed = len(gt_positions) - matched_count
    false = len(track_positions) - matched_count
    cardinality = (c**p / p) * (missed + false)
    value = (localization + cardinality) ** (1.0 / p)
    return float(value), matched


def _build_ground_truth(
    ground_truth: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """[evaluación] Prepara el GT por dron para interpolar.

    Pertenece a evaluación: organiza la referencia.
    """
    tracks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for drone_key, group in ground_truth.groupby("drone_key"):
        ordered = group.sort_values("ts_ms")
        times = ordered["ts_ms"].to_numpy(dtype=np.int64)
        positions = _positions(ordered)
        tracks[str(drone_key)] = (times, positions)
    return tracks


def _interpolate_ground_truth(
    tracks: Mapping[str, tuple[np.ndarray, np.ndarray]],
    timestamp: int,
) -> tuple[list[str], np.ndarray]:
    """[evaluación] Interpola el GT a un timestamp de track.

    Pertenece a evaluación: interpola la referencia.
    """
    keys: list[str] = []
    positions: list[np.ndarray] = []
    for drone_key, (times, values) in tracks.items():
        if timestamp < int(times[0]) or timestamp > int(times[-1]):
            continue
        keys.append(drone_key)
        positions.append(
            np.asarray(
                [
                    np.interp(timestamp, times, values[:, axis])
                    for axis in range(3)
                ],
                dtype=float,
            )
        )
    if not positions:
        return [], np.empty((0, 3), dtype=float)
    return keys, np.vstack(positions)


def _nearest_track(
    position: np.ndarray,
    track_positions: np.ndarray,
    track_ids: list[str],
    max_distance: float,
) -> str | None:
    """[evaluación] Busca el track más cercano a un punto GT.

    Pertenece a evaluación: se usa para recall e ID switches.
    """
    if len(track_positions) == 0:
        return None
    distances = np.linalg.norm(track_positions - position[None, :], axis=1)
    index = int(np.argmin(distances))
    if distances[index] > max_distance:
        return None
    return track_ids[index]


def evaluate(
    tracks: pd.DataFrame,
    ground_truth: pd.DataFrame,
    cfg: Any = None,
) -> dict[str, Any]:
    """[evaluación] Calcula las métricas por iteración.

    Pertenece a evaluación: compara tracks contra ground truth.
    """
    evaluation_cfg = _subcfg(cfg, "evaluation")
    c = float(evaluation_cfg.get("gospa_c_m", 100.0))
    p = float(evaluation_cfg.get("gospa_p", 2.0))
    match_distance = float(evaluation_cfg.get("match_distance_m", 50.0))
    groups: list[dict[str, Any]] = []
    # Cada iteración se evalúa contra su propio GT.
    for split_group, track_group in tracks.groupby("split_group", sort=False):
        gt_group = ground_truth[ground_truth["split_group"] == split_group]
        if gt_group.empty or track_group.empty:
            continue
        gt_by_drone = _build_ground_truth(gt_group)
        track_times = np.sort(track_group["ts_ms"].unique())
        gospa_values: list[float] = []
        squared_errors: list[float] = []
        matched_total = 0
        gt_total = 0
        track_total = 0
        false_total = 0
        id_switches = 0
        last_track: dict[str, str] = {}
        # Timestamp a timestamp: interpola GT, empareja y acumula.
        for timestamp in track_times:
            gt_keys, gt_positions = _interpolate_ground_truth(
                gt_by_drone, int(timestamp)
            )
            if not gt_keys:
                continue
            current_tracks = track_group[track_group["ts_ms"] == timestamp]
            track_positions = _positions(current_tracks)
            track_ids = current_tracks["track_id"].astype(str).tolist()
            value, matched = gospa(track_positions, gt_positions, c, p)
            gospa_values.append(value)
            squared_errors.extend([distance**2 for _, _, distance in matched])
            matched_total += len(matched)
            gt_total += len(gt_positions)
            track_total += len(track_positions)
            false_total += len(track_positions) - len(matched)
            # ID switches: track más cercano por dron y cambio de identidad.
            for gt_index, gt_key in enumerate(gt_keys):
                nearest = _nearest_track(
                    gt_positions[gt_index],
                    track_positions,
                    track_ids,
                    match_distance,
                )
                if nearest is None:
                    continue
                previous = last_track.get(gt_key)
                if previous is not None and previous != nearest:
                    id_switches += 1
                last_track[gt_key] = nearest
        if not gospa_values:
            continue
        groups.append(
            {
                "split_group": str(split_group),
                "gospa": float(np.mean(gospa_values)),
                "rmse": float(np.sqrt(np.mean(squared_errors)))
                if squared_errors
                else float("nan"),
                "recall": float(matched_total / gt_total) if gt_total else 0.0,
                "false_track_ratio": float(false_total / track_total)
                if track_total
                else 0.0,
                "id_switches": int(id_switches),
                "n_timestamps": int(len(gospa_values)),
            }
        )
    if not groups:
        return {
            "gospa": float("nan"),
            "rmse": float("nan"),
            "recall": 0.0,
            "false_track_ratio": 0.0,
            "id_switches": 0,
            "n_groups": 0,
            "groups": [],
        }
    return {
        "gospa": float(np.mean([row["gospa"] for row in groups])),
        "rmse": float(np.mean([row["rmse"] for row in groups])),
        "recall": float(np.mean([row["recall"] for row in groups])),
        "false_track_ratio": float(
            np.mean([row["false_track_ratio"] for row in groups])
        ),
        "id_switches": int(sum(row["id_switches"] for row in groups)),
        "n_groups": int(len(groups)),
        "groups": groups,
    }


def run_evaluation(
    tracks: pd.DataFrame,
    ground_truth: pd.DataFrame,
    cfg: Any,
) -> dict[str, Any]:
    """[evaluación] Calcula y escribe `reports/tracking_metrics.csv`.

    Pertenece a evaluación: es el cierre del pipeline.
    """
    result = evaluate(tracks, ground_truth, cfg)
    reports = cfg.reports if hasattr(cfg, "reports") else None
    if reports is not None:
        pd.DataFrame(result["groups"]).to_csv(
            reports / "tracking_metrics.csv", index=False
        )
    return result
