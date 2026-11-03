"""Conversiones geodésicas WGS84: ECEF, ENU, azimut y origen local.

Existe porque EGOI llega en radianes y el tracker necesita metros locales;
por eso convierte WGS84 a ENU con un origen por iteración.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

# Elipsoide WGS84: semieje mayor, aplanamiento y primera excentricidad al cuadrado.
# Referencia: NIMA TR8350.2, WGS84.
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def _array(value: float | Iterable[float] | np.ndarray) -> np.ndarray:
    # Convierte escalares o listas a ndarray float para vectorizar.
    return np.asarray(value, dtype=float)


def valid_latlon(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    # Rango geodésico válido: lat [-90, 90], lon [-180, 180].
    return (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= -90.0)
        & (lat <= 90.0)
        & (lon >= -180.0)
        & (lon <= 180.0)
    )


def geodetic_to_ecef(
    lat_deg: float | Iterable[float],
    lon_deg: float | Iterable[float],
    alt_m: float | Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Geodésico WGS84 (lat, lon, altura) -> ECEF en metros.
    # N es el radio de curvatura en el primer vertical.
    lat = np.radians(_array(lat_deg))
    lon = np.radians(_array(lon_deg))
    h = _array(alt_m)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + h) * cos_lat * np.cos(lon)
    y = (n + h) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * sin_lat
    return x, y, z


def ecef_to_geodetic(
    x: float | Iterable[float],
    y: float | Iterable[float],
    z: float | Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ECEF en metros -> geodésico WGS84.
    # Inversa iterativa; 6 iteraciones bastan para precisión sub-milimétrica.
    xa = _array(x)
    ya = _array(y)
    za = _array(z)
    lon = np.arctan2(ya, xa)
    p = np.hypot(xa, ya)
    lat = np.arctan2(za, p * (1.0 - WGS84_E2))
    h = np.zeros_like(lat)
    for _ in range(6):
        sin_lat = np.sin(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        cos_lat = np.cos(lat)
        safe_cos_lat = np.where(np.abs(cos_lat) < 1e-12, np.nan, cos_lat)
        h = p / safe_cos_lat - n
        lat = np.arctan2(za, p * (1.0 - WGS84_E2 * n / (n + h)))
    sin_lat = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    cos_lat = np.cos(lat)
    safe_cos_lat = np.where(np.abs(cos_lat) < 1e-12, np.nan, cos_lat)
    h = p / safe_cos_lat - n
    # Caso polar: p ~ 0; se evita dividir por cos(lat) ~ 0.
    pole = p < 1e-9
    lat = np.where(pole, np.sign(za) * (np.pi / 2.0), lat)
    h = np.where(pole, np.abs(za) - WGS84_B, h)
    return np.degrees(lat), np.degrees(lon), h


def geodetic_to_enu(
    lat_deg: float | Iterable[float],
    lon_deg: float | Iterable[float],
    alt_m: float | Iterable[float],
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convierte WGS84 a ENU local respecto a un origen."""
    # Geodésico -> ENU local (East, North, Up) en metros.
    # Se calcula la diferencia ECEF y se rota al plano tangente en el origen.
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
    dx = x - x0
    dy = y - y0
    dz = z - z0
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)
    sin_lon0 = np.sin(lon0)
    cos_lon0 = np.cos(lon0)
    east = -sin_lon0 * dx + cos_lon0 * dy
    north = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
    up = cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz
    return east, north, up


def enu_to_geodetic(
    east: float | Iterable[float],
    north: float | Iterable[float],
    up: float | Iterable[float],
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ENU local en metros -> geodésico WGS84.
    # Es la rotación inversa de geodetic_to_enu.
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)
    sin_lon0 = np.sin(lon0)
    cos_lon0 = np.cos(lon0)
    e = _array(east)
    n = _array(north)
    u = _array(up)
    dx = -sin_lon0 * e - sin_lat0 * cos_lon0 * n + cos_lat0 * cos_lon0 * u
    dy = cos_lon0 * e - sin_lat0 * sin_lon0 * n + cos_lat0 * sin_lon0 * u
    dz = cos_lat0 * n + sin_lat0 * u
    return ecef_to_geodetic(x0 + dx, y0 + dy, z0 + dz)


def normalize_azimuth(az_deg: float | Iterable[float]) -> np.ndarray:
    # Normaliza azimut a [0, 360).
    return np.mod(_array(az_deg), 360.0)


def angle_difference_deg(a_deg: float | Iterable[float], b_deg: float | Iterable[float]) -> np.ndarray:
    # Diferencia angular firmada en [-180, 180).
    return (np.asarray(a_deg, dtype=float) - np.asarray(b_deg, dtype=float) + 180.0) % 360.0 - 180.0


def median_origin(
    lat: Iterable[float],
    lon: Iterable[float],
    alt: Iterable[float],
    fallback: tuple[float, float, float],
) -> tuple[float, float, float, int]:
    """Calcula un origen robusto por mediana de posiciones válidas."""
    # Origen robusto por iteración: mediana de posiciones válidas.
    # Si no hay ninguna válida, usa el origen de emergencia.
    lat_arr = _array(lat)
    lon_arr = _array(lon)
    alt_arr = _array(alt)
    mask = valid_latlon(lat_arr, lon_arr) & np.isfinite(alt_arr)
    count = int(mask.sum())
    if count == 0:
        return float(fallback[0]), float(fallback[1]), float(fallback[2]), 0
    return (
        float(np.median(lat_arr[mask])),
        float(np.median(lon_arr[mask])),
        float(np.median(alt_arr[mask])),
        count,
    )


def rotation_ecef_to_enu(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Matriz de rotación ECEF -> ENU (filas East, North, Up)."""
    lat = np.radians(float(lat_deg))
    lon = np.radians(float(lon_deg))
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=float,
    )


def covariance_neu_to_neu(
    cov_neu: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    lat_deg: float,
    lon_deg: float,
) -> np.ndarray:
    """Rota una covarianza NEU del origen (lat0,lon0) a NEU en (lat,lon)."""
    cov = np.asarray(cov_neu, dtype=float)
    permutation = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    rot_origin = rotation_ecef_to_enu(lat0_deg, lon0_deg)
    rot_state = rotation_ecef_to_enu(lat_deg, lon_deg)
    rot = rot_state.T @ rot_origin
    cov_enu = permutation @ cov @ permutation.T
    cov_enu = rot @ cov_enu @ rot.T
    cov_neu = permutation @ cov_enu @ permutation.T
    return 0.5 * (cov_neu + cov_neu.T)
