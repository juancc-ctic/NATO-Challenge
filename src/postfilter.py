"""Postfiltro opcional de tracks.

Flujo: postfiltro, después del remuestreo. Ofrece relleno de huecos, suavizado
y descarte de tracks físicamente imposibles; no es gating, asociación ni
actualización, y delega en un método concreto seleccionado por configuración.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

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
    """[postfiltro] Aplica los pasos de postfiltro en orden.

    `postfilter.methods` es una lista; se ejecutan en secuencia sobre el mismo
    DataFrame. Si no existe, se acepta el antiguo `postfilter.method` de un solo
    paso por compatibilidad.
    """
    postfilter_cfg = _subcfg(tracker_cfg, "postfilter")
    methods = postfilter_cfg.get("methods")
    if methods is None:
        legacy = str(postfilter_cfg.get("method", "none"))
        methods = [] if legacy in ("", "none", "None") else [legacy]
    if isinstance(methods, str):
        methods = [methods]
    output = tracks
    for method in methods:
        method = str(method)
        if method in ("", "none", "None"):
            continue
        if method == "fill_gaps":
            output = _fill_gaps(output, tracker_cfg, grid_ms)
        elif method == "moving_average":
            output = _moving_average(output, tracker_cfg)
        elif method == "plausibility":
            output = _plausibility(output, tracker_cfg, grid_ms)
        elif method == "tube_merge":
            output = _tube_merge(output, tracker_cfg)
        else:
            raise ValueError(f"Método de postfiltro no soportado: {method}")
    return output


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
    for _, track in tracks.groupby(["split_group", "track_id"], sort=False):
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
            "sensor_class_hint",
            "sensor_serial_hint",
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
            output.groupby(["split_group", "track_id"], sort=False)[column]
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
    for _, track in tracks.groupby(["split_group", "track_id"], sort=False):
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


def _tube_merge(
    tracks: pd.DataFrame,
    tracker_cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """[postfiltro] Fusiona tracks que caen dentro del tubo de otro track.

    Aproxima el tubo por la unión de esferas de radio `radius_m` centradas en
    los puntos de la rejilla. Dos tracks se consideran el mismo si, en su
    intervalo temporal común, una fracción `min_overlap_ratio` del track corto
    queda dentro del tubo del otro. Se conserva un único representante por
    grupo de duplicados.
    """
    postfilter_cfg = _subcfg(tracker_cfg, "postfilter")
    merge_cfg = postfilter_cfg.get("tube_merge", {}) or {}
    radius_m = float(merge_cfg.get("radius_m", 25.0))
    min_ratio = float(merge_cfg.get("min_overlap_ratio", 0.8))
    min_overlap_s = float(merge_cfg.get("min_time_overlap_s", 2.0))
    min_points = int(merge_cfg.get("min_points", 5))
    keep_priority = str(merge_cfg.get("keep", "longest"))
    output: list[pd.DataFrame] = []
    for _, group in tracks.groupby("split_group", sort=False):
        ids = sorted(group["track_id"].astype(str).unique())
        stats: dict[str, dict[str, Any]] = {}
        for track_id in ids:
            track = group[group["track_id"].astype(str) == track_id].sort_values("ts_ms")
            times = track["ts_ms"].to_numpy(dtype=np.int64)
            positions = track[["n", "e", "u"]].to_numpy(dtype=float)
            finite = np.isfinite(positions).all(axis=1)
            times = times[finite]
            positions = positions[finite]
            if len(times) == 0:
                continue
            sigma = np.sqrt(
                np.clip(
                    track["C_nn"].to_numpy(dtype=float)
                    + track["C_ee"].to_numpy(dtype=float),
                    0.0,
                    None,
                )
            )
            sigma = sigma[finite]
            stats[track_id] = {
                "times": times,
                "positions": positions,
                "t0": int(times.min()),
                "t1": int(times.max()),
                "duration": (int(times.max()) - int(times.min())) / 1000.0,
                "hits": int(track["hits"].max()) if "hits" in track else len(track),
                "sigma": float(np.nanmedian(sigma)) if len(sigma) else float("inf"),
            }
        ids = [track_id for track_id in ids if track_id in stats and len(stats[track_id]["times"]) >= min_points]
        parent = {track_id: track_id for track_id in ids}

        def find(track_id: str) -> str:
            while parent[track_id] != track_id:
                parent[track_id] = parent[parent[track_id]]
                track_id = parent[track_id]
            return track_id

        def union(left: str, right: str) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for index, left in enumerate(ids):
            left_stats = stats[left]
            for right in ids[index + 1:]:
                right_stats = stats[right]
                start = max(left_stats["t0"], right_stats["t0"])
                end = min(left_stats["t1"], right_stats["t1"])
                if (end - start) < min_overlap_s * 1000.0:
                    continue
                left_mask = (left_stats["times"] >= start) & (left_stats["times"] <= end)
                right_mask = (right_stats["times"] >= start) & (right_stats["times"] <= end)
                left_points = left_stats["positions"][left_mask]
                right_points = right_stats["positions"][right_mask]
                if len(left_points) < min_points or len(right_points) < min_points:
                    continue
                if len(left_points) <= len(right_points):
                    distances, _ = cKDTree(right_points).query(left_points)
                else:
                    distances, _ = cKDTree(left_points).query(right_points)
                if float(np.mean(distances <= radius_m)) >= min_ratio:
                    union(left, right)

        clusters: dict[str, list[str]] = {}
        for track_id in ids:
            clusters.setdefault(find(track_id), []).append(track_id)
        keep_ids: set[str] = {track_id for track_id in ids if track_id not in stats}
        for members in clusters.values():
            if len(members) == 1:
                keep_ids.add(members[0])
                continue
            if keep_priority == "hits":
                best = max(members, key=lambda t: (stats[t]["hits"], stats[t]["duration"]))
            elif keep_priority == "lower_sigma":
                best = min(members, key=lambda t: (stats[t]["sigma"], -stats[t]["duration"]))
            else:
                best = max(members, key=lambda t: (stats[t]["duration"], stats[t]["hits"]))
            keep_ids.add(best)
        output.append(group[group["track_id"].astype(str).isin(keep_ids)])
    if not output:
        return pd.DataFrame(columns=TRACK_COLUMNS)
    return pd.concat(output, ignore_index=True)[TRACK_COLUMNS]
