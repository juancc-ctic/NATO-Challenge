"""Postfiltro opcional de tracks.

Flujo: postfiltro, después del remuestreo. Ofrece relleno de huecos, suavizado
y descarte de tracks físicamente imposibles; no es gating, asociación ni
actualización, y delega en un método concreto seleccionado por configuración.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.schema import TRACK_COLUMNS


def _subcfg(cfg: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    if not cfg:
        return {}
    value = cfg.get(key, {})
    return value if isinstance(value, Mapping) else {}


def postprocess_tracks(
    tracks: pd.DataFrame,
    tracker_cfg: Mapping[str, Any] | None,
    grid_ms: int,
) -> pd.DataFrame:
    """[postfiltro] Aplica el postfiltro configurado.

    Pertenece a postfiltro, después del remuestreo; por defecto no cambia nada y
    delega en el método concreto.
    """
    method = str(_subcfg(tracker_cfg, "postfilter").get("method", "none"))
    if method == "none":
        return tracks
    if method == "fill_gaps":
        return _fill_gaps(tracks, tracker_cfg, grid_ms)
    if method == "moving_average":
        return _moving_average(tracks, tracker_cfg)
    if method == "plausibility":
        return _plausibility(tracks, tracker_cfg, grid_ms)
    raise ValueError(f"Método de postfiltro no soportado: {method}")


def _fill_gaps(
    tracks: pd.DataFrame,
    tracker_cfg: Mapping[str, Any],
    grid_ms: int,
) -> pd.DataFrame:
    """[postfiltro] Rellena huecos por interpolación lineal.

    Pertenece a postfiltro: interpola observaciones ausentes dentro de un
    track.
    """
    postfilter_cfg = _subcfg(tracker_cfg, "postfilter")
    max_gap_ms = int(postfilter_cfg.get("max_gap_ms", 1000))
    limit = max(1, max_gap_ms // max(grid_ms, 1))
    output: list[pd.DataFrame] = []
    for _, track in tracks.groupby("track_id", sort=False):
        ordered = track.sort_values("ts_ms").set_index("ts_ms")
        start = int(ordered.index.min())
        end = int(ordered.index.max())
        # Reindexa a la rejilla y rellena solo huecos internos.
        index = np.arange(start, end + 1, grid_ms, dtype=np.int64)
        filled = ordered.reindex(index)
        for column in ("n", "e", "u"):
            filled[column] = filled[column].interpolate(
                method="linear", limit=limit, limit_area="inside"
            )
        # Covarianza y metadatos se arrastran; no se inventan.
        for column in ("C_nn", "C_ne", "C_nu", "C_ee", "C_eu", "C_uu"):
            filled[column] = filled[column].ffill()
        for column in (
            "split_group",
            "track_id",
            "status",
            "hits",
            "misses",
            "age",
            "last_update_ms",
            "class_norm",
            "serial_norm",
            "classification",
            "classification_confidence",
            "identity",
        ):
            filled[column] = filled[column].ffill()
        filled = filled.dropna(subset=["n", "e"])
        filled = filled.reset_index().rename(columns={"index": "ts_ms"})
        output.append(filled)
    if not output:
        return pd.DataFrame(columns=TRACK_COLUMNS)
    return pd.concat(output, ignore_index=True)[TRACK_COLUMNS]


def _moving_average(
    tracks: pd.DataFrame,
    tracker_cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """[postfiltro] Suaviza posiciones con media móvil centrada.

    Pertenece a postfiltro: es una mejora opcional del track final.
    """
    window = int(_subcfg(tracker_cfg, "postfilter").get("smooth_window", 5))
    if window <= 1:
        return tracks
    if window % 2 == 0:
        window += 1
    # Media móvil centrada por track, sin mezclar tracks distintos.
    output = tracks.copy()
    for column in ("n", "e", "u"):
        output[column] = (
            output.groupby("track_id", sort=False)[column]
            .transform(
                lambda values: values.rolling(
                    window=window, center=True, min_periods=1
                ).mean()
            )
        )
    return output[TRACK_COLUMNS]


def _plausibility(
    tracks: pd.DataFrame,
    tracker_cfg: Mapping[str, Any],
    grid_ms: int,
) -> pd.DataFrame:
    """[postfiltro] Descarta tracks con velocidad imposible.

    Pertenece a postfiltro: es una comprobación opcional de plausibilidad
    física.
    """
    max_speed_mps = float(
        _subcfg(tracker_cfg, "postfilter").get("max_speed_mps", 90.0)
    )
    step_s = max(grid_ms, 1) / 1000.0
    keep: list[pd.DataFrame] = []
    for _, track in tracks.groupby("track_id", sort=False):
        ordered = track.sort_values("ts_ms")
        positions = ordered[["n", "e", "u"]].to_numpy(dtype=float)
        if len(positions) < 2:
            keep.append(ordered)
            continue
        # Velocidad entre muestras consecutivas de la rejilla.
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        speeds = steps / step_s
        # Si algún salto supera v_max, se descarta el track completo.
        if float(np.nanmax(speeds)) <= max_speed_mps:
            keep.append(ordered)
    if not keep:
        return pd.DataFrame(columns=TRACK_COLUMNS)
    return pd.concat(keep, ignore_index=True)[TRACK_COLUMNS]
