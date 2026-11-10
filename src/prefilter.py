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
