"""Prefiltro de detecciones antes del tracker.

Flujo: prefiltro → predicción → gating → asociación → actualización →
ciclo de vida → remuestreo → postfiltro. Limpia detecciones inválidas y aplica
un método concreto de plausibilidad seleccionado por configuración.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def _subcfg(cfg: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    if not cfg:
        return {}
    value = cfg.get(key, {})
    return value if isinstance(value, Mapping) else {}


def preprocess_detections(
    detections: pd.DataFrame,
    tracker_cfg: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """[prefiltro] Limpia y filtra detecciones antes del tracker.

    Aplica la limpieza común (fusión, no-dron y medidas válidas) y delega en el
    método concreto de prefiltro (`range` o `none`). No trackea ni asocia.
    """
    frame = _clean_detections(detections, tracker_cfg)
    method = str(_subcfg(tracker_cfg, "prefilter").get("method", "range"))
    if method == "none":
        return frame
    if method == "range":
        return _preprocess_range(frame, tracker_cfg)
    raise ValueError(f"Método de prefiltro no soportado: {method}")


def _clean_detections(
    detections: pd.DataFrame,
    tracker_cfg: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """[prefiltro] Limpieza común de detecciones.

    Elimina nodos de fusión, no-drones y detecciones sin medida utilizable.
    """
    prefilter_cfg = _subcfg(tracker_cfg, "prefilter")
    # Limpieza común: fusión, no-dron y medidas utilizables.
    frame = detections.copy()
    if bool(prefilter_cfg.get("drop_fusion", True)):
        if "is_fusion_node" in frame.columns:
            frame = frame[~frame["is_fusion_node"].fillna(False).astype(bool)]
        frame = frame[frame["sensor_type"].astype(str) != "fusion"]
    if bool(prefilter_cfg.get("drop_non_drone", True)):
        if "class_is_non_drone" in frame.columns:
            frame = frame[~frame["class_is_non_drone"].fillna(False).astype(bool)]
    drop_classes = {str(value) for value in prefilter_cfg.get("drop_class_norm", []) or []}
    if drop_classes:
        frame = frame[~frame["class_norm"].astype(str).isin(drop_classes)]
    drop_by_sensor = prefilter_cfg.get("drop_class_norm_by_sensor", {}) or {}
    for sensor_type, classes in drop_by_sensor.items():
        classes_set = {str(value) for value in classes or []}
        if not classes_set:
            continue
        mask = (
            frame["sensor_type"].astype(str) == str(sensor_type)
        ) & frame["class_norm"].astype(str).isin(classes_set)
        frame = frame[~mask]
    drop_sensor_types = {
        str(value) for value in prefilter_cfg.get("drop_sensor_types", []) or []
    }
    if drop_sensor_types:
        frame = frame[~frame["sensor_type"].astype(str).isin(drop_sensor_types)]
    keep_sensor_types = {
        str(value) for value in prefilter_cfg.get("keep_sensor_types", []) or []
    }
    if keep_sensor_types:
        frame = frame[frame["sensor_type"].astype(str).isin(keep_sensor_types)]
    min_det_conf = prefilter_cfg.get("min_det_conf")
    if min_det_conf is not None:
        frame = frame[frame["det_conf"].fillna(0.0) >= float(min_det_conf)]
    min_class_conf = prefilter_cfg.get("min_class_conf")
    if min_class_conf is not None:
        has_class = frame["has_class"].fillna(False).astype(bool)
        frame = frame[~has_class | (frame["class_conf"].fillna(0.0) >= float(min_class_conf))]
    for sensor_type, value in (
        prefilter_cfg.get("min_det_conf_by_sensor", {}) or {}
    ).items():
        mask = (
            frame["sensor_type"].astype(str) == str(sensor_type)
        ) & (frame["det_conf"].fillna(0.0) < float(value))
        frame = frame[~mask]
    for sensor_type, value in (
        prefilter_cfg.get("max_det_conf_by_sensor", {}) or {}
    ).items():
        mask = (
            frame["sensor_type"].astype(str) == str(sensor_type)
        ) & (frame["det_conf"].fillna(0.0) > float(value))
        frame = frame[~mask]
    max_alt_m = prefilter_cfg.get("max_alt_m")
    min_alt_m = prefilter_cfg.get("min_alt_m")
    if max_alt_m is not None or min_alt_m is not None:
        position_rows = frame["has_position"].fillna(False).astype(bool)
        altitude = frame["u"]
        altitude_ok = pd.Series(True, index=frame.index)
        if max_alt_m is not None:
            altitude_ok &= altitude.isna() | (altitude <= float(max_alt_m))
        if min_alt_m is not None:
            altitude_ok &= altitude.isna() | (altitude >= float(min_alt_m))
        frame = frame[~position_rows | altitude_ok]
    position_ok = (
        frame["has_position"].fillna(False).astype(bool)
        & frame["n"].notna()
        & frame["e"].notna()
    )
    bearing_ok = (
        frame["has_bearing"].fillna(False).astype(bool)
        & frame["az_deg"].notna()
        & frame["has_sensor_location"].fillna(False).astype(bool)
    )
    return frame[position_ok | bearing_ok]


def _preprocess_range(
    detections: pd.DataFrame,
    tracker_cfg: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """[prefiltro] Descarta ecos de posición fuera de rango.

    Es un filtro de plausibilidad espacial: elimina medidas de posición a más de
    `max_range_m` del origen ENU. Las medidas solo-azimut se conservan.
    """
    max_range_m = float(
        _subcfg(tracker_cfg, "prefilter").get("max_range_m", 5000.0)
    )
    if detections.empty:
        return detections
    # Solo aplica el radio a filas con posición; el azimut no tiene rango.
    position_rows = detections["has_position"].fillna(False).astype(bool)
    range_ok = np.hypot(detections["n"], detections["e"]) <= max_range_m
    return detections[~position_rows | range_ok]
