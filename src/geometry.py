"""Modelos de medida, información y diagnóstico geométrico.

Flujo: gating/actualización (modelos `H` y `R`), prefiltro/diagnóstico
(selección de sensores) y diagnóstico/mejora (`J = Σ HᵀR⁻¹H`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

POSITION_SENSOR_TYPES = {"radar", "cyber", "fusion", "other"}
DEFAULT_SIGMA_AZ_DEG = {"rf": 5.0, "acoustic": 10.0, "camera": 1.0}
DEFAULT_SIGMA_M = 50.0


@dataclass(frozen=True)
class Measurement:
    """[gating/actualización] Medida linealizada de un sensor.

    Pertenece a gating/actualización: empaqueta `z`, `H` y `R` para que
    gating calcule el residuo y la innovación, y actualización la ganancia.
    """

    z: np.ndarray
    H: np.ndarray
    R: np.ndarray
    sensor_id: str
    sensor_type: str
    class_norm: str | None = None
    serial_norm: str | None = None


@dataclass(frozen=True)
class GeometryDiagnostics:
    """[diagnóstico/mejora] Resumen de observabilidad de `J`.

    Pertenece a diagnóstico/mejora: indica rango, condición y log-determinante
    de la información acumulada.
    """

    eigenvalues: np.ndarray
    rank: int
    condition: float
    logdet: float


def _finite(value: Any) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any, default: float) -> float:
    return float(value) if _finite(value) else float(default)


def _subcfg(cfg: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    if not cfg:
        return {}
    value = cfg.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _sigma_az_deg(detection: Mapping[str, Any], sensor_type: str, cfg: Mapping[str, Any] | None) -> float:
    value = detection.get("sigma_az_deg")
    if _finite(value):
        return float(value)
    defaults = _subcfg(cfg, "measurement").get("sigma_az_deg", {})
    if isinstance(defaults, Mapping) and sensor_type in defaults:
        return float(defaults[sensor_type])
    return float(DEFAULT_SIGMA_AZ_DEG.get(sensor_type, 5.0))


def _position_measurement(
    detection: Mapping[str, Any],
    state: np.ndarray,
    cfg: Mapping[str, Any] | None,
) -> Measurement | None:
    """[gating/actualización] Modelo de medida de posición.

    Construye `z`, `H` y `R`; no es asociación ni ciclo de vida.
    """
    state_dim = int(state.size)
    if state_dim < 2 or not _finite(detection.get("n")) or not _finite(detection.get("e")):
        return None
    use_altitude = (
        state_dim >= 3
        and bool((cfg or {}).get("use_altitude", True))
        and _finite(detection.get("u"))
        and _finite(detection.get("sigma_u"))
    )
    rows = 3 if use_altitude else 2
    H = np.zeros((rows, state_dim), dtype=float)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    z = [float(detection["n"]), float(detection["e"])]
    sigmas = [
        _as_float(detection.get("sigma_n"), DEFAULT_SIGMA_M),
        _as_float(detection.get("sigma_e"), DEFAULT_SIGMA_M),
    ]
    if use_altitude:
        H[2, 2] = 1.0
        z.append(float(detection["u"]))
        sigmas.append(float(detection["sigma_u"]))
    return Measurement(
        z=np.asarray(z, dtype=float),
        H=H,
        R=np.diag(np.square(np.asarray(sigmas, dtype=float))),
        sensor_id=str(detection.get("sensor_id", "")),
        sensor_type=str(detection.get("sensor_type", "other")),
        class_norm=detection.get("class_norm"),
        serial_norm=detection.get("serial_norm"),
    )


def _azimuth_measurement(
    detection: Mapping[str, Any],
    state: np.ndarray,
    cfg: Mapping[str, Any] | None,
) -> Measurement | None:
    """[gating/actualización] Modelo de medida de azimut.

    Construye `z`, `H` y `R` con el Jacobiano no lineal; no es asociación.
    """
    state_dim = int(state.size)
    sensor_n = detection.get("sensor_n")
    sensor_e = detection.get("sensor_e")
    if (
        state_dim < 2
        or not _finite(detection.get("az_deg"))
        or not _finite(sensor_n)
        or not _finite(sensor_e)
    ):
        return None
    dn = float(state[0]) - float(sensor_n)
    de = float(state[1]) - float(sensor_e)
    r2 = dn * dn + de * de
    if r2 <= 1e-9:
        return None
    sensor_type = str(detection.get("sensor_type", "other"))
    sigma_az_rad = math.radians(_sigma_az_deg(detection, sensor_type, cfg))
    H = np.zeros((1, state_dim), dtype=float)
    H[0, 0] = -de / r2
    H[0, 1] = dn / r2
    z = [math.radians(float(detection["az_deg"]))]
    sigmas = [sigma_az_rad]
    use_elevation = (
        state_dim >= 3
        and bool(_subcfg(cfg, "measurement").get("use_elevation", False))
        and _finite(detection.get("el_deg"))
        and _finite(detection.get("sensor_u"))
    )
    if use_elevation:
        sensor_u = float(detection["sensor_u"])
        du = float(state[2]) - sensor_u
        rho2 = r2 + du * du
        rho = math.sqrt(rho2)
        r = math.sqrt(r2)
        H_el = np.zeros((1, state_dim), dtype=float)
        H_el[0, 0] = -du * dn / (rho2 * r)
        H_el[0, 1] = -du * de / (rho2 * r)
        H_el[0, 2] = r / rho2
        H = np.vstack([H, H_el])
        z.append(math.radians(float(detection["el_deg"])))
        sigma_el_deg = _subcfg(cfg, "measurement").get("sigma_el_deg", sigma_az_rad)
        sigmas.append(math.radians(float(sigma_el_deg)))
    return Measurement(
        z=np.asarray(z, dtype=float),
        H=H,
        R=np.diag(np.square(np.asarray(sigmas, dtype=float))),
        sensor_id=str(detection.get("sensor_id", "")),
        sensor_type=sensor_type,
        class_norm=detection.get("class_norm"),
        serial_norm=detection.get("serial_norm"),
    )


def measurement_model(
    detection: Mapping[str, Any],
    state: np.ndarray,
    cfg: Mapping[str, Any] | None = None,
) -> Measurement | None:
    """[gating/actualización] Construye `z`, `H` y `R`.

    Pertenece a gating y actualización: es el modelo de medida que usan para
    el residuo/innovación y para la ganancia de Kalman.
    """
    predicted = np.asarray(state, dtype=float).reshape(-1)
    sensor_type = str(detection.get("sensor_type", "other"))
    has_position = bool(detection.get("has_position", False))
    has_bearing = bool(detection.get("has_bearing", False))
    if has_position and sensor_type in POSITION_SENSOR_TYPES:
        return _position_measurement(detection, predicted, cfg)
    if has_bearing:
        return _azimuth_measurement(detection, predicted, cfg)
    return None


def measurement_information(measurement: Measurement) -> np.ndarray:
    """[diagnóstico/mejora] Información que aporta una medida: `HᵀR⁻¹H`.

    Pertenece a diagnóstico/mejora: es la contribución geométrica que se suma
    para analizar observabilidad y redundancia.
    """
    return measurement.H.T @ np.linalg.inv(measurement.R) @ measurement.H


def information_matrix(measurements: Sequence[Measurement]) -> np.ndarray:
    """[diagnóstico/mejora] Suma la información: `J = Σ HᵀR⁻¹H`.

    Pertenece a diagnóstico/mejora: es la matriz que se usa para diagnosticar
    si la geometría permite observar la posición.
    """
    if not measurements:
        raise ValueError("Se necesita al menos una medida")
    state_dim = int(measurements[0].H.shape[1])
    J = np.zeros((state_dim, state_dim), dtype=float)
    for measurement in measurements:
        J = J + measurement_information(measurement)
    return J


def _logdet_psd(matrix: np.ndarray, rank_tolerance: float) -> float:
    """[diagnóstico/mejora] Log-determinante estable de una PSD.

    Se usa en la selección de sensores; no es filtro ni gating.
    """
    values = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    if values.size == 0:
        return 0.0
    scale = max(1.0, float(values.max()))
    values = np.maximum(values, max(rank_tolerance * scale, 1e-12))
    return float(np.sum(np.log(values)))


def diagnose_information(
    J: np.ndarray,
    rank_tolerance: float = 1e-9,
) -> GeometryDiagnostics:
    """[diagnóstico/mejora] Calcula rango, condición y log-determinante de `J`.

    Pertenece a diagnóstico/mejora: es una comprobación de observabilidad
    geométrica sobre la información acumulada.
    """
    eigenvalues = np.linalg.eigvalsh(np.asarray(J, dtype=float))
    eigenvalues = np.sort(eigenvalues)[::-1]
    if eigenvalues.size == 0:
        return GeometryDiagnostics(
            eigenvalues=eigenvalues,
            rank=0,
            condition=float("inf"),
            logdet=float("-inf"),
        )
    max_eigenvalue = float(eigenvalues[0])
    tolerance = max(max_eigenvalue * rank_tolerance, 1e-12)
    rank = int(np.sum(eigenvalues > tolerance))
    min_eigenvalue = float(eigenvalues[-1])
    condition = max_eigenvalue / min_eigenvalue if min_eigenvalue > tolerance else float("inf")
    logdet = (
        float(np.sum(np.log(eigenvalues[eigenvalues > tolerance])))
        if rank > 0
        else float("-inf")
    )
    return GeometryDiagnostics(
        eigenvalues=eigenvalues,
        rank=rank,
        condition=condition,
        logdet=logdet,
    )


def select_sensors(
    measurements: Sequence[Measurement],
    prior_covariance: np.ndarray | None = None,
    max_sensors: int | None = None,
    method: str = "greedy_logdet",
    rank_tolerance: float = 1e-9,
) -> list[int]:
    """[prefiltro/diagnóstico] Selección de sensores por `logdet`.

    Puede usarse como prefiltro opcional antes del tracker: elige un subconjunto
    de sensores maximizando `logdet` de la información acumulada. También sirve
    como diagnóstico de redundancia y observabilidad.
    """
    if method != "greedy_logdet":
        raise ValueError(f"Método de selección no soportado: {method}")
    if not measurements:
        return []
    state_dim = int(measurements[0].H.shape[1])
    J = np.zeros((state_dim, state_dim), dtype=float)
    if prior_covariance is not None:
        J = np.linalg.inv(np.asarray(prior_covariance, dtype=float))
    remaining = list(range(len(measurements)))
    selected: list[int] = []
    current = _logdet_psd(J, rank_tolerance)
    while remaining and (max_sensors is None or len(selected) < max_sensors):
        best_index = -1
        best_gain = -np.inf
        best_J = J
        for index in remaining:
            candidate = J + measurement_information(measurements[index])
            gain = _logdet_psd(candidate, rank_tolerance) - current
            if gain > best_gain:
                best_index = index
                best_gain = gain
                best_J = candidate
        if best_index < 0 or best_gain <= 0.0:
            break
        selected.append(best_index)
        remaining.remove(best_index)
        J = best_J
        current = _logdet_psd(J, rank_tolerance)
    return selected
