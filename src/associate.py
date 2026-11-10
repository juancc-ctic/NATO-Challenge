"""Gating y asignación de detecciones a tracks.

Flujo: gating → asociación. El gating descarta asociaciones improbables y la
asociación resuelve el problema uno-a-uno; ambos delegan en métodos concretos
(χ² y Húngaro).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from src.geometry import Measurement

DEFAULT_INVALID_COST = 1.0e12


@dataclass(frozen=True)
class AssociationResult:
    """[asociación] Pares asignados y elementos sin asignar.

    Pertenece a asociación: es la salida del paso de asociación.
    """

    pairs: list[tuple[int, int]]
    unassigned_tracks: list[int]
    unassigned_detections: list[int]


def _subcfg(cfg: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    if not cfg:
        return {}
    value = cfg.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _known(value: Any) -> bool:
    """[asociación] Indica si una etiqueta es utilizable.

    Se usa para regularizar el coste por clase/serial; no es gating.
    """
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {
        "unknown",
        "nan",
        "none",
        "null",
        "not applicable",
        "n/a",
    }


def _chi2_threshold(dof: int, cfg: Mapping[str, Any] | None) -> float:
    """[gating] Umbral χ² según confianza y grados de libertad.

    Pertenece a gating: fija la puerta que usa `_gating_chi2`.
    """
    gating_cfg = _subcfg(cfg, "gating")
    fixed = gating_cfg.get("chi2_gate")
    if fixed is not None:
        return float(fixed)
    confidence = float(gating_cfg.get("confidence", 0.999))
    return float(chi2.ppf(confidence, dof))


def gating(
    track_states: Sequence[np.ndarray],
    track_covariances: Sequence[np.ndarray],
    detections: Sequence[Mapping[str, Any]],
    measurement_fn: Callable[[Mapping[str, Any], np.ndarray], Measurement | None],
    method: str = "chi2",
    cfg: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """[gating] Devuelve máscara válida y distancia Mahalanobis².

    Pertenece a gating, antes de la asociación; delega en el método concreto
    (χ²) y devuelve la máscara válida y la distancia Mahalanobis².
    """
    if method == "chi2":
        return _gating_chi2(
            track_states,
            track_covariances,
            detections,
            measurement_fn,
            cfg,
        )
    raise ValueError(f"Método de gating no soportado: {method}")


def _gating_chi2(
    track_states: Sequence[np.ndarray],
    track_covariances: Sequence[np.ndarray],
    detections: Sequence[Mapping[str, Any]],
    measurement_fn: Callable[[Mapping[str, Any], np.ndarray], Measurement | None],
    cfg: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """[gating] Implementación χ² concreta.

    Calcula residuo, innovación y Mahalanobis²; no es asociación.
    """
    states = [np.asarray(state, dtype=float).reshape(-1) for state in track_states]
    covariances = [np.asarray(cov, dtype=float) for cov in track_covariances]
    n_tracks = len(states)
    n_detections = len(detections)
    # Matriz T tracks x D detecciones.
    valid = np.zeros((n_tracks, n_detections), dtype=bool)
    distances = np.full((n_tracks, n_detections), np.inf, dtype=float)
    if n_tracks == 0 or n_detections == 0:
        return valid, distances
    gate_distance_m = _subcfg(cfg, "gating").get("gate_distance_m")
    for i, (state, covariance) in enumerate(zip(states, covariances)):
        for j, detection in enumerate(detections):
            measurement = measurement_fn(detection, state)
            if measurement is None:
                continue
            # Residuo e innovación para la distancia de Mahalanobis.
            residual = measurement.z - measurement.H @ state
            innovation = measurement.H @ covariance @ measurement.H.T + measurement.R
            try:
                distance = float(residual.T @ np.linalg.solve(innovation, residual))
            except np.linalg.LinAlgError:
                continue
            distances[i, j] = distance
            accepted = distance <= _chi2_threshold(int(measurement.z.size), cfg)
            # Puerta física opcional para medidas de posición.
            if gate_distance_m is not None and measurement.z.size >= 2:
                position_distance = float(
                    np.linalg.norm(measurement.z[:2] - state[:2])
                )
                accepted = accepted and position_distance <= float(gate_distance_m)
            valid[i, j] = accepted
    return valid, distances


def build_cost_matrix(
    distances: np.ndarray,
    valid: np.ndarray,
    track_classes: Sequence[Any],
    track_serials: Sequence[Any],
    detections: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """[asociación] Construye el coste con regularización de clase/serial.

    Pertenece a asociación: las penalizaciones de clase y serial son la
    regularización del coste; el gating ya marcó las entradas inválidas.
    """
    association_cfg = _subcfg(cfg, "association")
    invalid_cost = float(association_cfg.get("invalid_cost", DEFAULT_INVALID_COST))
    class_weight = float(association_cfg.get("cost_class_weight", 0.0))
    serial_weight = float(association_cfg.get("cost_serial_weight", 0.0))
    cost = np.asarray(distances, dtype=float).copy()
    cost[~np.asarray(valid, dtype=bool)] = invalid_cost
    # Penaliza clase/serial distintos; el gating ya invalidó pares imposibles.
    for i in range(cost.shape[0]):
        for j, detection in enumerate(detections):
            if not bool(valid[i, j]):
                continue
            detection_class = detection.get("class_norm")
            if (
                class_weight > 0.0
                and _known(detection_class)
                and _known(track_classes[i])
                and detection_class != track_classes[i]
            ):
                cost[i, j] += class_weight
            detection_serial = detection.get("serial_norm")
            if (
                serial_weight > 0.0
                and _known(detection_serial)
                and _known(track_serials[i])
                and detection_serial != track_serials[i]
            ):
                cost[i, j] += serial_weight
    return cost


def assignment(
    cost: np.ndarray,
    valid: np.ndarray,
    method: str = "hungarian",
    cfg: Mapping[str, Any] | None = None,
) -> AssociationResult:
    """[asociación] Resuelve la asignación uno-a-uno.

    Pertenece a asociación; delega en el método concreto (Húngaro).
    """
    if method == "hungarian":
        return _assignment_hungarian(cost, valid, cfg)
    raise ValueError(f"Método de asignación no soportado: {method}")


def _assignment_hungarian(
    cost: np.ndarray,
    valid: np.ndarray,
    cfg: Mapping[str, Any] | None,
) -> AssociationResult:
    """[asociación] Implementación Húngara concreta.

    Pertenece a asociación: resuelve el problema uno-a-uno sobre el coste.
    """
    cost_matrix = np.asarray(cost, dtype=float)
    valid_mask = np.asarray(valid, dtype=bool)
    if cost_matrix.shape != valid_mask.shape:
        raise ValueError("El coste y la máscara válida deben tener la misma forma")
    n_tracks, n_detections = cost_matrix.shape
    if n_tracks == 0 or n_detections == 0:
        return AssociationResult(
            pairs=[],
            unassigned_tracks=list(range(n_tracks)),
            unassigned_detections=list(range(n_detections)),
        )
    invalid_cost = float(
        _subcfg(cfg, "association").get("invalid_cost", DEFAULT_INVALID_COST)
    )
    # Coste grande para pares inválidos; Húngaro no los elegirá.
    masked = np.where(
        valid_mask,
        np.nan_to_num(cost_matrix, nan=invalid_cost, posinf=invalid_cost),
        invalid_cost,
    )
    rows, cols = linear_sum_assignment(masked)
    # Solo se conservan los pares válidos; el resto queda sin asignar.
    pairs = [
        (int(row), int(col))
        for row, col in zip(rows, cols)
        if bool(valid_mask[row, col])
    ]
    assigned_tracks = {row for row, _ in pairs}
    assigned_detections = {col for _, col in pairs}
    return AssociationResult(
        pairs=pairs,
        unassigned_tracks=[
            index for index in range(n_tracks) if index not in assigned_tracks
        ],
        unassigned_detections=[
            index for index in range(n_detections) if index not in assigned_detections
        ],
    )
