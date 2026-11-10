"""Tracker EKF multi-sensor con ciclo de vida.

Flujo: predicción → gating → asociación → actualización → ciclo de vida →
remuestreo → postfiltro. Predicción y actualización son el filtro; gating y
asociación se delegan en `src/associate.py`; ciclo de vida, remuestreo y
postfiltro son capas de gestión posteriores.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.associate import assignment, build_cost_matrix, gating
from src.geometry import measurement_model
from src.motion import motion_model
from src.postfilter import postprocess_tracks
from src.prefilter import preprocess_detections
from src.resample import resample_tracks
from src.schema import TRACK_COLUMNS


@dataclass
class Track:
    """[ciclo de vida] Estado interno de un track.

    Pertenece al ciclo de vida: guarda estado, covarianza, contadores y
    etiquetas que se usan para confirmar, coast o eliminar.
    """

    track_id: str
    split_group: str
    state: np.ndarray
    covariance: np.ndarray
    status: str = "tentative"
    hits: int = 1
    misses: int = 0
    age: int = 1
    last_update_ms: int = 0
    class_norm: str | None = None
    serial_norm: str | None = None


def _tracker_config(cfg: Any) -> Mapping[str, Any]:
    """[orquestación] Extrae la configuración del tracker.

    Pertenece a orquestación: solo resuelve parámetros.
    """
    if hasattr(cfg, "data"):
        return cfg.data.get("tracker", {}) or {}
    return cfg if isinstance(cfg, Mapping) else {}


def _grid_ms(cfg: Any) -> int:
    """[remuestreo] Lee la rejilla temporal de entrega.

    Pertenece a remuestreo: solo fija los timestamps de salida.
    """
    if hasattr(cfg, "data"):
        return int(cfg.data.get("submission", {}).get("grid_ms", 100))
    if isinstance(cfg, Mapping):
        return int(cfg.get("output_grid_ms", 100))
    return 100


def _subcfg(cfg: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    if not cfg:
        return {}
    value = cfg.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _finite(value: Any) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any, default: float) -> float:
    return float(value) if _finite(value) else float(default)


def _is_true(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def predict(track: Track, dt: float, tracker_cfg: Mapping[str, Any]) -> None:
    """[predicción] Predice estado y covarianza del track.

    Pertenece a predicción del EKF: aplica `F` y `Q` antes de gating y
    asociación.
    """
    method = str(_subcfg(tracker_cfg, "motion").get("model", "constant_velocity"))
    F, Q = motion_model(dt, method=method, cfg=tracker_cfg)
    track.state = F @ track.state
    track.covariance = F @ track.covariance @ F.T + Q


def update(
    track: Track,
    measurement: Any,
    now_ms: int,
    tracker_cfg: Mapping[str, Any],
) -> None:
    """[actualización] Corrige el track con una medida linealizada.

    Pertenece a actualización del EKF: calcula innovación, ganancia de Kalman y
    covarianza con la forma de Joseph.
    """
    residual = measurement.z - measurement.H @ track.state
    innovation = measurement.H @ track.covariance @ measurement.H.T + measurement.R
    gain = track.covariance @ measurement.H.T @ np.linalg.inv(innovation)
    track.state = track.state + gain @ residual
    identity = np.eye(track.state.size)
    correction = identity - gain @ measurement.H
    track.covariance = (
        correction @ track.covariance @ correction.T
        + gain @ measurement.R @ gain.T
    )
    track.last_update_ms = int(now_ms)
    if measurement.class_norm is not None:
        track.class_norm = measurement.class_norm
    if measurement.serial_norm is not None:
        track.serial_norm = measurement.serial_norm


def initialize_track(
    detection: Mapping[str, Any],
    track_id: str,
    split_group: str,
    now_ms: int,
    tracker_cfg: Mapping[str, Any],
) -> Track:
    """[ciclo de vida] Crea un track tentative desde una detección.

    Pertenece al ciclo de vida: es el nacimiento de un track y fija estado y
    covarianza iniciales.
    """
    state_dim = int(tracker_cfg.get("state_dim", 3))
    if state_dim not in (2, 3):
        raise ValueError("state_dim debe ser 2 o 3")
    measurement_cfg = _subcfg(tracker_cfg, "measurement")
    altitude_sigma = float(measurement_cfg.get("initial_altitude_sigma_m", 1000.0))
    velocity_sigma = float(measurement_cfg.get("initial_velocity_sigma_mps", 50.0))
    state = np.zeros(2 * state_dim, dtype=float)
    state[0] = float(detection["n"])
    state[1] = float(detection["e"])
    covariance = np.zeros((2 * state_dim, 2 * state_dim), dtype=float)
    covariance[0, 0] = _as_float(detection.get("sigma_n"), 50.0) ** 2
    covariance[1, 1] = _as_float(detection.get("sigma_e"), 50.0) ** 2
    if state_dim == 3:
        state[2] = _as_float(detection.get("u"), 0.0)
        covariance[2, 2] = _as_float(detection.get("sigma_u"), altitude_sigma) ** 2
    covariance[state_dim:, state_dim:] = (velocity_sigma**2) * np.eye(state_dim)
    return Track(
        track_id=track_id,
        split_group=split_group,
        state=state,
        covariance=covariance,
        status="tentative",
        hits=1,
        misses=0,
        age=1,
        last_update_ms=int(now_ms),
        class_norm=detection.get("class_norm"),
        serial_norm=detection.get("serial_norm"),
    )


def _can_initiate(
    detection: Mapping[str, Any],
    tracker_cfg: Mapping[str, Any] | None = None,
) -> bool:
    """[ciclo de vida] Decide si una detección puede iniciar un track.

    Pertenece al ciclo de vida: decide nacimiento de un track.
    """
    if (
        not _is_true(detection.get("has_position"))
        or not _finite(detection.get("n"))
        or not _finite(detection.get("e"))
        or _is_true(detection.get("class_is_non_drone"))
    ):
        return False
    if bool(
        _subcfg(tracker_cfg, "lifecycle").get(
            "initiate_from_drone_only", False
        )
    ):
        return _is_true(detection.get("class_is_drone"))
    return True


def _track_priority(track: Track) -> tuple[int, int]:
    """[ciclo de vida] Ordena tracks para fusionar duplicados.

    Pertenece al ciclo de vida: es una regla de gestión de tracks.
    """
    status_rank = {"confirmed": 2, "tentative": 1, "coast": 0, "deleted": -1}
    return (status_rank.get(track.status, -1), track.hits)


def _filter_initiations(
    detections: Sequence[Mapping[str, Any]],
    tracker_cfg: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """[ciclo de vida] Filtra detecciones para iniciar tracks.

    Pertenece al ciclo de vida: exige coincidencia multisensor para iniciar.
    """
    lifecycle_cfg = _subcfg(tracker_cfg, "lifecycle")
    min_sensors = int(lifecycle_cfg.get("initiation_min_sensors", 1))
    if min_sensors <= 1:
        return list(detections)
    distance_m = float(lifecycle_cfg.get("initiation_distance_m", 100.0))
    filtered: list[Mapping[str, Any]] = []
    for index, detection in enumerate(detections):
        # Un track con clase de dron puede iniciarse con un solo sensor.
        if _is_true(detection.get("class_is_drone")):
            filtered.append(detection)
            continue
        if not _finite(detection.get("n")) or not _finite(detection.get("e")):
            continue
        # Un track desconocido necesita varios sensores dentro del radio.
        sensors = {str(detection.get("sensor_id", ""))}
        for other_index, other in enumerate(detections):
            if index == other_index:
                continue
            if str(other.get("sensor_id", "")) == str(
                detection.get("sensor_id", "")
            ):
                continue
            if not _finite(other.get("n")) or not _finite(other.get("e")):
                continue
            distance = math.hypot(
                float(detection["n"]) - float(other["n"]),
                float(detection["e"]) - float(other["e"]),
            )
            if distance <= distance_m:
                sensors.add(str(other.get("sensor_id", "")))
        if len(sensors) >= min_sensors:
            filtered.append(detection)
    return filtered


def _merge_tracks(tracks: list[Track], merge_distance_m: float) -> list[Track]:
    """[ciclo de vida] Fusiona tracks duplicados cercanos.

    Pertenece al ciclo de vida: es gestión y fusión de tracks.
    """
    if merge_distance_m <= 0.0 or len(tracks) < 2:
        return tracks
    # Prioriza confirmed > tentative > coast y conserva el track con más hits.
    ordered = sorted(tracks, key=_track_priority, reverse=True)
    for i, track in enumerate(ordered):
        if track.status == "deleted":
            continue
        for other in ordered[i + 1 :]:
            if other.status == "deleted":
                continue
            distance = float(np.linalg.norm(track.state[:2] - other.state[:2]))
            if distance > merge_distance_m:
                continue
            if (
                _known(track.serial_norm)
                and _known(other.serial_norm)
                and track.serial_norm != other.serial_norm
            ):
                continue
            if (
                _known(track.class_norm)
                and _known(other.class_norm)
                and track.class_norm != other.class_norm
            ):
                continue
            other.status = "deleted"
    return [track for track in tracks if track.status != "deleted"]


def _known(value: Any) -> bool:
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


def manage_tracks(
    tracks: list[Track],
    updated_ids: set[str],
    unassigned_detections: Sequence[Mapping[str, Any]],
    now_ms: int,
    tracker_cfg: Mapping[str, Any],
    next_id: int,
    split_group: str,
) -> tuple[list[Track], int]:
    """[ciclo de vida] Confirma, coast, elimina y crea tracks.

    Pertenece al ciclo de vida: es el paso de gestión de tracks.
    """
    lifecycle_cfg = _subcfg(tracker_cfg, "lifecycle")
    hits_to_confirm = int(lifecycle_cfg.get("hits_to_confirm", 3))
    max_coast_ms = int(float(lifecycle_cfg.get("max_coast_s", 5.0)) * 1000)
    max_tentative_misses = int(lifecycle_cfg.get("max_tentative_misses", 2))
    merge_distance_m = float(lifecycle_cfg.get("merge_distance_m", 50.0))
    for track in tracks:
        if track.track_id in updated_ids:
            track.hits += 1
            track.misses = 0
            track.age += 1
            if track.status in {"tentative", "coast"} and track.hits >= hits_to_confirm:
                track.status = "confirmed"
        else:
            track.misses += 1
            track.age += 1
            if track.status == "confirmed":
                track.status = "coast"
            if now_ms - track.last_update_ms > max_coast_ms:
                track.status = "deleted"
            elif track.status == "tentative" and track.misses >= max_tentative_misses:
                track.status = "deleted"
    tracks = [track for track in tracks if track.status != "deleted"]
    for detection in _filter_initiations(unassigned_detections, tracker_cfg):
        if not _can_initiate(detection, tracker_cfg):
            continue
        track_id = f"{split_group}:T{next_id:04d}"
        next_id += 1
        tracks.append(
            initialize_track(
                detection,
                track_id,
                split_group,
                now_ms,
                tracker_cfg,
            )
        )
    return _merge_tracks(tracks, merge_distance_m), next_id


def _snapshot(track: Track, now_ms: int) -> dict[str, Any]:
    """[ciclo de vida] Guarda una instantánea del track.

    Pertenece al ciclo de vida: guarda la instantánea que luego usa el
    remuestreo.
    """
    state_dim = int(track.state.size // 2)
    return {
        "split_group": track.split_group,
        "ts_ms": int(now_ms),
        "track_id": track.track_id,
        "state": track.state.copy(),
        "covariance": track.covariance.copy(),
        "status": track.status,
        "hits": track.hits,
        "misses": track.misses,
        "age": track.age,
        "last_update_ms": track.last_update_ms,
        "class_norm": track.class_norm,
        "serial_norm": track.serial_norm,
        "u": float(track.state[2]) if state_dim >= 3 else float("nan"),
    }


def run_tracker(detections: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """[orquestación] Ejecuta el flujo completo del tracker.

    Flujo: prefiltro → predicción → gating → asociación → actualización →
    ciclo de vida → remuestreo → postfiltro. Devuelve tracks en la rejilla.
    """
    tracker_cfg = _tracker_config(cfg)
    grid_ms = _grid_ms(cfg)
    output_tentative = bool(tracker_cfg.get("output_tentative", True))
    gating_method = str(_subcfg(tracker_cfg, "gating").get("method", "chi2"))
    assignment_method = str(
        _subcfg(tracker_cfg, "association").get("method", "hungarian")
    )
    # Prefiltro: limpia y descarta detecciones no plausibles.
    frame = preprocess_detections(detections, tracker_cfg)
    if frame.empty:
        return pd.DataFrame(columns=TRACK_COLUMNS)
    frame = frame.sort_values(["split_group", "ts_ms", "sensor_id"])
    output: list[pd.DataFrame] = []
    # Cada split_group es una iteración independiente.
    for split_group, group in frame.groupby("split_group", sort=False):
        timestamps = np.sort(group["ts_ms"].unique())
        tracks: list[Track] = []
        history: list[dict[str, Any]] = []
        next_id = 1
        last_timestamp: int | None = None
        for timestamp in timestamps:
            current = group[group["ts_ms"] == timestamp]
            if tracks and last_timestamp is not None:
                # Predice todos los tracks al nuevo instante.
                dt = (int(timestamp) - int(last_timestamp)) / 1000.0
                if dt > 0.0:
                    for track in tracks:
                        predict(track, dt, tracker_cfg)
                updated_ids: set[str] = set()
                unassigned: list[Mapping[str, Any]] = []
                # Asocia por sensor para permitir fusión multisensor.
                for _, sensor_detections in current.groupby("sensor_id", sort=False):
                    records = sensor_detections.to_dict("records")
                    track_states = [track.state for track in tracks]
                    track_covariances = [track.covariance for track in tracks]
                    measurement_fn = _make_measurement_fn(tracker_cfg)
                    # Gating χ²: descarta pares track-detección improbables.
                    valid, distances = gating(
                        track_states,
                        track_covariances,
                        records,
                        measurement_fn,
                        method=gating_method,
                        cfg=tracker_cfg,
                    )
                    # Coste = Mahalanobis + penalización de clase/serial.
                    cost = build_cost_matrix(
                        distances,
                        valid,
                        [track.class_norm for track in tracks],
                        [track.serial_norm for track in tracks],
                        records,
                        cfg=tracker_cfg,
                    )
                    # Húngaro: asignación uno-a-uno entre tracks y detecciones.
                    result = assignment(
                        cost,
                        valid,
                        method=assignment_method,
                        cfg=tracker_cfg,
                    )
                    # Actualiza los tracks que han recibido detección.
                    for track_index, detection_index in result.pairs:
                        measurement = measurement_fn(
                            records[detection_index],
                            tracks[track_index].state,
                        )
                        if measurement is None:
                            continue
                        update(
                            tracks[track_index],
                            measurement,
                            int(timestamp),
                            tracker_cfg,
                        )
                        updated_ids.add(tracks[track_index].track_id)
                    # Las detecciones libres pasan al ciclo de vida.
                    for detection_index in result.unassigned_detections:
                        unassigned.append(records[detection_index])
                # Ciclo de vida: confirma, coast, elimina y crea tracks.
                tracks, next_id = manage_tracks(
                    tracks,
                    updated_ids,
                    unassigned,
                    int(timestamp),
                    tracker_cfg,
                    next_id,
                    str(split_group),
                )
            else:
                # Primer instante: crea tracks desde detecciones válidas.
                records = current.to_dict("records")
                unassigned = [
                    record
                    for record in records
                    if _can_initiate(record, tracker_cfg)
                ]
                tracks, next_id = manage_tracks(
                    [],
                    set(),
                    unassigned,
                    int(timestamp),
                    tracker_cfg,
                    next_id,
                    str(split_group),
                )
            # Guarda el estado interno para el remuestreo final.
            for track in tracks:
                history.append(_snapshot(track, int(timestamp)))
            last_timestamp = int(timestamp)
        # Remuestreo a la rejilla de entrega y postfiltro opcional.
        resampled = resample_tracks(history, grid_ms, tracker_cfg)
        resampled = postprocess_tracks(resampled, tracker_cfg, grid_ms)
        if not output_tentative and not resampled.empty:
            resampled = resampled[resampled["status"] != "tentative"]
        output.append(resampled)
    if not output:
        return pd.DataFrame(columns=TRACK_COLUMNS)
    return pd.concat(output, ignore_index=True)


def _make_measurement_fn(tracker_cfg: Mapping[str, Any]) -> Any:
    """[orquestación] Crea la función de medida para gating/actualización.

    Pertenece a orquestación: delega el modelo en `measurement_model`.
    """
    def build(detection: Mapping[str, Any], state: np.ndarray) -> Any:
        return measurement_model(detection, state, tracker_cfg)

    return build
