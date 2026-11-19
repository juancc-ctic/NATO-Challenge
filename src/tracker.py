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
from src.geometry import measurement_model, measurement_residual, sigma_az_deg
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
    initiated_by_drone: bool = False
    bearing_only: bool = False


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
    if measurement.kind == "bearing" and track.bearing_only:
        residual = measurement_residual(measurement)
    else:
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
    if getattr(measurement, "kind", "position") == "position":
        track.bearing_only = False
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
    covariance = np.zeros((2 * state_dim, 2 * state_dim), dtype=float)
    has_position = (
        _is_true(detection.get("has_position"))
        and _finite(detection.get("n"))
        and _finite(detection.get("e"))
    )
    if has_position:
        state[0] = float(detection["n"])
        state[1] = float(detection["e"])
        covariance[0, 0] = _as_float(detection.get("sigma_n"), 50.0) ** 2
        covariance[1, 1] = _as_float(detection.get("sigma_e"), 50.0) ** 2
        if state_dim == 3:
            state[2] = _as_float(detection.get("u"), 0.0)
            covariance[2, 2] = _as_float(detection.get("sigma_u"), altitude_sigma) ** 2
    else:
        _initialize_bearing_state(
            state, covariance, detection, state_dim, tracker_cfg
        )
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
        initiated_by_drone=_is_true(detection.get("class_is_drone")),
        bearing_only=not has_position,
    )


def _initialize_bearing_state(
    state: np.ndarray,
    covariance: np.ndarray,
    detection: Mapping[str, Any],
    state_dim: int,
    tracker_cfg: Mapping[str, Any],
) -> None:
    """[ciclo de vida] Inicia un estado 3D desde solo azimut.

    Usa un prior de rango y de altitud para romper la ambigüedad de escala;
    la covarianza es ancha en radial y estrecha en tangencial.
    """
    lifecycle_cfg = _subcfg(tracker_cfg, "lifecycle")
    measurement_cfg = _subcfg(tracker_cfg, "measurement")
    prior_range = float(lifecycle_cfg.get("bearing_init_range_m", 1000.0))
    range_sigma = float(
        lifecycle_cfg.get("bearing_init_range_sigma_m", 1000.0)
    )
    prior_altitude = float(lifecycle_cfg.get("bearing_init_altitude_m", 50.0))
    altitude_sigma = float(
        measurement_cfg.get("initial_altitude_sigma_m", 1000.0)
    )
    az = math.radians(float(detection["az_deg"]))
    sensor_n = float(detection["sensor_n"])
    sensor_e = float(detection["sensor_e"])
    state[0] = sensor_n + prior_range * math.cos(az)
    state[1] = sensor_e + prior_range * math.sin(az)
    if state_dim >= 3:
        state[2] = prior_altitude
    sigma_az = math.radians(
        sigma_az_deg(
            detection,
            str(detection.get("sensor_type", "other")),
            tracker_cfg,
        )
    )
    sigma_tangent = prior_range * sigma_az
    radial = np.array([math.cos(az), math.sin(az)], dtype=float)
    tangent = np.array([-math.sin(az), math.cos(az)], dtype=float)
    covariance[:2, :2] = (
        range_sigma**2 * np.outer(radial, radial)
        + sigma_tangent**2 * np.outer(tangent, tangent)
    )
    if state_dim >= 3:
        covariance[2, 2] = altitude_sigma**2


def _is_bearing_only(detection: Mapping[str, Any]) -> bool:
    """[ciclo de vida] Indica si una detección solo aporta azimut.

    Pertenece al ciclo de vida: exige azimut y posición del sensor, no del
    objeto.
    """
    return (
        not _is_true(detection.get("has_position"))
        and _is_true(detection.get("has_bearing"))
        and _finite(detection.get("az_deg"))
        and _finite(detection.get("sensor_n"))
        and _finite(detection.get("sensor_e"))
    )


def _can_initiate(
    detection: Mapping[str, Any],
    tracker_cfg: Mapping[str, Any] | None = None,
) -> bool:
    """[ciclo de vida] Decide si una detección puede iniciar un track.

    Pertenece al ciclo de vida: decide nacimiento de un track.
    """
    if _is_true(detection.get("class_is_non_drone")):
        return False
    has_position = (
        _is_true(detection.get("has_position"))
        and _finite(detection.get("n"))
        and _finite(detection.get("e"))
    )
    lifecycle_cfg = _subcfg(tracker_cfg, "lifecycle")
    if has_position:
        if bool(lifecycle_cfg.get("initiate_from_drone_only", False)):
            return _is_true(detection.get("class_is_drone"))
        return True
    if not bool(lifecycle_cfg.get("initiate_from_bearing", False)):
        return False
    return _is_bearing_only(detection)


def _track_priority(track: Track) -> tuple[int, int]:
    """[ciclo de vida] Ordena tracks para fusionar duplicados.

    Pertenece al ciclo de vida: es una regla de gestión de tracks.
    """
    status_rank = {"confirmed": 2, "tentative": 1, "coast": 0, "deleted": -1}
    return (status_rank.get(track.status, -1), track.hits)


def _filter_initiations(
    detections: Sequence[Mapping[str, Any]],
    buffer: Sequence[Mapping[str, Any]],
    tracker_cfg: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """[ciclo de vida] Filtra detecciones para iniciar tracks.

    Pertenece al ciclo de vida: exige coincidencia multisensor dentro de una
    ventana temporal y un radio, no en el mismo timestamp exacto.
    """
    lifecycle_cfg = _subcfg(tracker_cfg, "lifecycle")
    min_sensors = int(lifecycle_cfg.get("initiation_min_sensors", 1))
    initiate_from_bearing = bool(
        lifecycle_cfg.get("initiate_from_bearing", False)
    )
    if min_sensors <= 1 and not initiate_from_bearing:
        return list(detections)
    distance_m = float(lifecycle_cfg.get("initiation_distance_m", 100.0))
    window_ms = int(float(lifecycle_cfg.get("initiation_window_s", 1.0)) * 1000)
    filtered: list[Mapping[str, Any]] = []
    for detection in detections:
        has_position = (
            _is_true(detection.get("has_position"))
            and _finite(detection.get("n"))
            and _finite(detection.get("e"))
        )
        if has_position:
            # Un track con clase de dron puede iniciarse con un solo sensor.
            if _is_true(detection.get("class_is_drone")):
                filtered.append(detection)
                continue
            if not _finite(detection.get("ts_ms")):
                continue
            # Un track desconocido necesita varios sensores dentro de la ventana.
            sensors = {str(detection.get("sensor_id", ""))}
            for other in buffer:
                if str(other.get("sensor_id", "")) == str(
                    detection.get("sensor_id", "")
                ):
                    continue
                if not _finite(other.get("n")) or not _finite(
                    other.get("e")
                ):
                    continue
                if not _finite(other.get("ts_ms")):
                    continue
                if (
                    abs(int(detection["ts_ms"]) - int(other["ts_ms"]))
                    > window_ms
                ):
                    continue
                distance = math.hypot(
                    float(detection["n"]) - float(other["n"]),
                    float(detection["e"]) - float(other["e"]),
                )
                if distance <= distance_m:
                    sensors.add(str(other.get("sensor_id", "")))
            if len(sensors) >= min_sensors:
                filtered.append(detection)
        elif initiate_from_bearing and _is_bearing_only(detection):
            if _bearing_supported(detection, buffer, lifecycle_cfg):
                filtered.append(detection)
    return filtered


def _bearing_supported(
    detection: Mapping[str, Any],
    buffer: Sequence[Mapping[str, Any]],
    lifecycle_cfg: Mapping[str, Any],
) -> bool:
    """[ciclo de vida] Exige varios bearings coherentes en el tiempo.

    Pertenece al ciclo de vida: corrobora un inicio solo-azimut con el mismo
    sensor, evitando depender de una única detección.
    """
    min_bearings = int(lifecycle_cfg.get("bearing_init_min_bearings", 3))
    if min_bearings <= 1:
        return True
    window_ms = int(
        float(lifecycle_cfg.get("bearing_init_window_s", 2.0)) * 1000
    )
    tolerance = math.radians(
        float(lifecycle_cfg.get("bearing_init_angle_deg", 10.0))
    )
    if not _finite(detection.get("ts_ms")):
        return False
    az0 = math.radians(float(detection["az_deg"]))
    ts0 = int(detection["ts_ms"])
    sensor_id = str(detection.get("sensor_id", ""))
    times: set[int] = set()
    for other in buffer:
        if not _is_bearing_only(other):
            continue
        if str(other.get("sensor_id", "")) != sensor_id:
            continue
        ts = int(other["ts_ms"])
        if abs(ts - ts0) > window_ms:
            continue
        az = math.radians(float(other["az_deg"]))
        delta = abs((az - az0 + math.pi) % (2.0 * math.pi) - math.pi)
        if delta <= tolerance:
            times.add(ts)
    return len(times) >= min_bearings


def _update_initiation_buffer(
    buffer: list[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    now_ms: int,
    tracker_cfg: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """[ciclo de vida] Mantiene detecciones recientes para corroborar inicios."""
    lifecycle_cfg = _subcfg(tracker_cfg, "lifecycle")
    window_ms = int(float(lifecycle_cfg.get("initiation_window_s", 1.0)) * 1000)
    if bool(lifecycle_cfg.get("initiate_from_bearing", False)):
        window_ms = max(
            window_ms,
            int(
                float(lifecycle_cfg.get("bearing_init_window_s", 2.0))
                * 1000
            ),
        )
    for detection in detections:
        if _can_initiate(detection, tracker_cfg):
            buffer.append(detection)
    return [
        detection
        for detection in buffer
        if _finite(detection.get("ts_ms"))
        and now_ms - int(detection["ts_ms"]) <= window_ms
    ]


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
            if bool(track.bearing_only) != bool(other.bearing_only):
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
    initiation_buffer: Sequence[Mapping[str, Any]],
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
    hits_to_confirm_unknown = int(
        lifecycle_cfg.get("hits_to_confirm_unknown", hits_to_confirm)
    )
    max_coast_ms = int(float(lifecycle_cfg.get("max_coast_s", 5.0)) * 1000)
    max_tentative_misses = int(lifecycle_cfg.get("max_tentative_misses", 2))
    merge_distance_m = float(lifecycle_cfg.get("merge_distance_m", 50.0))
    for track in tracks:
        if track.track_id in updated_ids:
            track.hits += 1
            track.misses = 0
            track.age += 1
            threshold = (
                hits_to_confirm
                if track.initiated_by_drone
                else hits_to_confirm_unknown
            )
            if track.status in {"tentative", "coast"} and track.hits >= threshold:
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
    for detection in _filter_initiations(
        unassigned_detections, initiation_buffer, tracker_cfg
    ):
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
        initiation_buffer: list[Mapping[str, Any]] = []
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
                        track_bearing_only=[
                            track.bearing_only for track in tracks
                        ],
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
                initiation_buffer = _update_initiation_buffer(
                    initiation_buffer,
                    unassigned,
                    int(timestamp),
                    tracker_cfg,
                )
                tracks, next_id = manage_tracks(
                    tracks,
                    updated_ids,
                    unassigned,
                    initiation_buffer,
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
                initiation_buffer = _update_initiation_buffer(
                    initiation_buffer,
                    unassigned,
                    int(timestamp),
                    tracker_cfg,
                )
                tracks, next_id = manage_tracks(
                    [],
                    set(),
                    unassigned,
                    initiation_buffer,
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
