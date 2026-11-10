"""Remuestreo de tracks a la rejilla de entrega.

Flujo: remuestreo, antes del postfiltro. Predice cada track con `F` y `Q`
sobre una rejilla temporal fija; no corrige con medidas ni interpola
detecciones.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.motion import motion_model
from src.schema import TRACK_COLUMNS


def _subcfg(cfg: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    if not cfg:
        return {}
    value = cfg.get(key, {})
    return value if isinstance(value, Mapping) else {}


def resample_tracks(
    history: Sequence[Mapping[str, Any]],
    grid_ms: int,
    tracker_cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """[remuestreo] Predice cada track sobre una rejilla fija.

    Pertenece a remuestreo: usa `F` y `Q` para llevar cada track a los
    timestamps de entrega.
    """
    # Agrupa instantáneas por track para remuestrear cada uno por separado.
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in history:
        grouped[(str(row["split_group"]), str(row["track_id"]))].append(row)
    output: list[dict[str, Any]] = []
    method = str(_subcfg(tracker_cfg, "motion").get("model", "constant_velocity"))
    for (split_group, track_id), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["ts_ms"]))
        times = np.asarray([int(row["ts_ms"]) for row in ordered], dtype=np.int64)
        first_ts = int(times[0])
        last_ts = int(times[-1])
        # Rejilla alineada a múltiplos de grid_ms que cubre la vida del track.
        start = int(math.floor(first_ts / grid_ms) * grid_ms)
        end = int(math.ceil(last_ts / grid_ms) * grid_ms)
        for timestamp in range(start, end + 1, grid_ms):
            # Última instantánea conocida en o antes del timestamp.
            index = int(np.searchsorted(times, timestamp, side="right") - 1)
            if index < 0:
                continue
            snapshot = ordered[index]
            state = np.asarray(snapshot["state"], dtype=float).copy()
            covariance = np.asarray(snapshot["covariance"], dtype=float).copy()
            dt = (timestamp - int(snapshot["ts_ms"])) / 1000.0
            if dt > 0.0:
                # Sin medida nueva: solo propagación con F y Q.
                F, Q = motion_model(dt, method=method, cfg=tracker_cfg)
                state = F @ state
                covariance = F @ covariance @ F.T + Q
            state_dim = int(state.size // 2)
            # Fila de salida: posición, covarianza y estado del track.
            output.append(
                {
                    "split_group": split_group,
                    "ts_ms": int(timestamp),
                    "track_id": track_id,
                    "n": float(state[0]),
                    "e": float(state[1]),
                    "u": float(state[2]) if state_dim >= 3 else float("nan"),
                    "C_nn": float(covariance[0, 0]),
                    "C_ne": float(covariance[0, 1]),
                    "C_nu": float(covariance[0, 2]) if state_dim >= 3 else 0.0,
                    "C_ee": float(covariance[1, 1]),
                    "C_eu": float(covariance[1, 2]) if state_dim >= 3 else 0.0,
                    "C_uu": float(covariance[2, 2]) if state_dim >= 3 else 0.0,
                    "status": snapshot["status"],
                    "hits": int(snapshot["hits"]),
                    "misses": int(snapshot["misses"]),
                    "age": int(snapshot["age"]),
                    "last_update_ms": int(snapshot["last_update_ms"]),
                    "class_norm": snapshot.get("class_norm"),
                    "serial_norm": snapshot.get("serial_norm"),
                    "classification": snapshot.get("class_norm") or "Unknown",
                    "classification_confidence": 0.0,
                    "identity": "Unknown",
                }
            )
    return pd.DataFrame(output, columns=TRACK_COLUMNS)
