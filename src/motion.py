"""Modelo de movimiento para predicción y remuestreo.

Flujo: predicción. Construye `F` (modelo de movimiento) y `Q` (ruido del
proceso); lo usan el tracker y el remuestreo, no el gating ni la asociación.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _subcfg(cfg: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    if not cfg:
        return {}
    value = cfg.get(key, {})
    return value if isinstance(value, Mapping) else {}


def motion_model(
    dt: float,
    method: str = "constant_velocity",
    cfg: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """[predicción] Devuelve `F` y `Q`.

    Pertenece a predicción; delega en el modelo concreto de movimiento
    (velocidad constante).
    """
    if method == "constant_velocity":
        return _constant_velocity(dt, cfg)
    raise ValueError(f"Modelo de movimiento no soportado: {method}")


def _constant_velocity(
    dt: float,
    cfg: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """[predicción] Modelo de velocidad constante con ruido DWNA.

    Pertenece a predicción: construye `F` y `Q`.
    """
    state_dim = int((cfg or {}).get("state_dim", 3))
    if state_dim not in (2, 3):
        raise ValueError("state_dim debe ser 2 o 3")
    step = float(dt)
    if step <= 0.0:
        size = 2 * state_dim
        return np.eye(size), np.zeros((size, size), dtype=float)
    max_dt = _subcfg(cfg, "motion").get("max_dt_s")
    if max_dt is not None:
        step = min(step, float(max_dt))
    identity = np.eye(state_dim)
    size = 2 * state_dim
    # F: posición += velocidad·Δt; velocidad se mantiene.
    F = np.eye(size, dtype=float)
    F[:state_dim, state_dim:] = step * identity
    acceleration = float(
        _subcfg(cfg, "motion").get("process_noise_mps2", 2.0)
    )
    variance = acceleration * acceleration
    # Q: ruido DWNA, varianza de aceleración integrada en Δt.
    Q = variance * np.block(
        [
            [(step**4 / 4.0) * identity, (step**3 / 2.0) * identity],
            [(step**3 / 2.0) * identity, (step**2) * identity],
        ]
    )
    return F, Q
