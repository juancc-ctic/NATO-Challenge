"""
Carga y normalización de los CSV de sensores (Delta_*) y de las trazas GNSS
de los drones (DroneTag_*).

Todo se lleva a un sistema local ENU (Este, Norte, Arriba) en metros, con
origen en el centro de la zona de operación, para que el gráfico 3D tenga
proporciones reales.
"""
from __future__ import annotations

import ast
import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

R_EARTH = 6_371_000.0


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _parse_list(s):
    """Convierte las celdas tipo "[{'type': ..., 'confidence': ...}]" en listas."""
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else [v]
    except (ValueError, SyntaxError):
        return []


def _classification(s):
    """Devuelve (clase, subclase, confianza) de la clase con mayor confianza."""
    items = _parse_list(s)
    if not items:
        return "", "", np.nan
    best = max(items, key=lambda d: d.get("confidence", -1) if isinstance(d, dict) else -1)
    cls = best.get("type", "")
    conf = best.get("confidence", np.nan)
    sub = ""
    subs = best.get("subClass") or []
    if subs and isinstance(subs[0], dict):
        sub = subs[0].get("type", "")
    return cls, sub, float(conf) if conf is not None else np.nan


def _to_deg(x, y, coord_system):
    """Pasa a grados si el sistema de coordenadas viene en radianes."""
    if isinstance(coord_system, str) and "RAD" in coord_system:
        return np.degrees(x), np.degrees(y)
    return x, y


@dataclass
class Projection:
    lon0: float
    lat0: float

    def enu(self, lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        e = np.radians(lon - self.lon0) * R_EARTH * np.cos(np.radians(self.lat0))
        n = np.radians(lat - self.lat0) * R_EARTH
        return e, n


# --------------------------------------------------------------------------- #
# Estructuras de salida
# --------------------------------------------------------------------------- #
@dataclass
class Sensor:
    key: str            # p.ej. "NYTF · RADAR"
    node: str           # NYTF
    kind: str           # RADAR
    file: str
    bearing_only: bool  # True si sólo da marcación (azimut / elevación)
    conf_source: str    # de dónde sale la confianza
    pos_e: float = 0.0
    pos_n: float = 0.0
    pos_z_raw: float = 0.0
    pos_z_agl: float = 0.0
    z_offset: float = 0.0   # offset vertical aplicado a las detecciones
    z_note: str = ""
    df: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class Drone:
    key: str      # p.ej. "RED · Mavic-3"
    name: str
    team: str
    file: str
    df: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class Dataset:
    sensors: dict
    drones: dict
    proj: Projection
    ground_ref: float
    t0: pd.Timestamp
    t1: pd.Timestamp


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
import re

_SENSOR_RE = re.compile(r"(Delta|Charlie)_([A-Z0-9]+)_(.+?)_\d{6}_\d{4}_\d{4}")
_DRONE_RE = re.compile(r"DroneTag_([A-Z]+)_\d{6}_\d{4}_\d{4}_(.+?)(?:-\d+)?\.csv$")


def _read_drone(path, proj=None):
    m = _DRONE_RE.search(os.path.basename(path))
    team, name = (m.group(1), m.group(2)) if m else ("?", os.path.basename(path))
    df = pd.read_csv(path, index_col=0)
    df["t"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("t").reset_index(drop=True)
    df["speed"] = np.sqrt(df["velocity_x"] ** 2 + df["velocity_y"] ** 2 + df["velocity_z"] ** 2)
    return Drone(key=f"{team} · {name}", name=name, team=team, file=os.path.basename(path), df=df)


def _read_sensor(path):
    base = os.path.basename(path)
    m = _SENSOR_RE.search(base)
    prefix, node, kind = (m.group(1), m.group(2), m.group(3)) if m else ("", base, "?")
    df = pd.read_csv(path, index_col=0)
    df["t"] = pd.to_datetime(df["timestamp"])

    # Posición del sensor (en grados)
    slon, slat = _to_deg(df["sensor_location_x"].astype(float),
                            df["sensor_location_y"].astype(float),
                            df["sensor_location_coordinateSystem"].iloc[0])
    df["s_lon"], df["s_lat"] = slon, slat
    df["s_z"] = df["sensor_location_z"].astype(float)

    # Clasificación
    cl = df["classification"].map(_classification)
    df["cls"] = [c[0] for c in cl]
    df["subcls"] = [c[1] for c in cl]
    df["cls_conf"] = [c[2] for c in cl]

    # Confianza: se usa la columna que realmente aporta información
    det = df["detectionConfidence"].astype(float) if "detectionConfidence" in df else None
    if det is not None and det.nunique() > 1:
        df["conf"], src = det, "detectionConfidence"
    elif df["cls_conf"].notna().any() and df["cls_conf"].nunique() > 1:
        df["conf"], src = df["cls_conf"], "confianza de clasificación"
    elif det is not None:
        df["conf"], src = det, "detectionConfidence (constante)"
    elif df["cls_conf"].notna().any():
        df["conf"], src = df["cls_conf"], "confianza de clasificación (constante)"
    else:
        df["conf"], src = np.nan, "no informada"
    df["conf_missing"] = df["conf"].isna()
    df["conf"] = df["conf"].fillna(1.0).clip(0, 1)

    bearing_only = "location_x" not in df.columns
    if not bearing_only:
        lon, lat = _to_deg(df["location_x"].astype(float), df["location_y"].astype(float),
                           df["location_coordinateSystem"].iloc[0])
        df["lon"], df["lat"] = lon, lat
        df["z_raw"] = df["location_z"].astype(float)
    else:
        df["az"] = df["rangeBearing_azimuth"].astype(float)
        df["el"] = df["rangeBearing_elevation"].astype(float) if "rangeBearing_elevation" in df else 0.0

    obj_col = "objectId" if "objectId" in df else None
    df["obj"] = df[obj_col].astype(str) if obj_col else ""
    if "state" in df:
        df["state"] = df["state"].fillna("")
    else:
        df["state"] = ""

    df = df.sort_values("t").reset_index(drop=True)
    return Sensor(key=f"{prefix} · {node} · {kind}" if prefix else f"{node} · {kind}",
                    node=node, kind=kind, file=base,
                    bearing_only=bearing_only, conf_source=src, df=df)


def _estimate_z_offset(sensor: Sensor, drones: dict, max_dist=25.0, max_dt=1.0, min_matches=15):
    """
    Estima el desfase vertical entre las detecciones de un sensor y la altitud
    elipsoidal GNSS de los drones (geo_altitude), emparejando detecciones
    cercanas en tiempo (<1 s) y en planta (<25 m).
    """
    d = sensor.df
    if sensor.bearing_only or d.empty:
        return None, 0
    allr = pd.concat([dr.df[["t", "e", "n", "geo_altitude"]].assign(drone=k) for k, dr in drones.items()])
    allr = allr.sort_values("t").reset_index(drop=True)
    tt = allr["t"].values.astype("datetime64[ns]").astype(np.int64) / 1e9
    sample = d.sample(min(len(d), 4000), random_state=0)
    st = sample["t"].values.astype("datetime64[ns]").astype(np.int64) / 1e9
    diffs = []
    for (te, e, n, z) in zip(st, sample["e"].values, sample["n"].values, sample["z_raw"].values):
        lo, hi = np.searchsorted(tt, [te - max_dt, te + max_dt])
        if hi <= lo:
            continue
        sub = allr.iloc[lo:hi]
        dist = np.hypot(sub["e"].values - e, sub["n"].values - n)
        i = dist.argmin()
        if dist[i] < max_dist:
            diffs.append((sub["drone"].values[i], sub["geo_altitude"].values[i] - z))
    if len(diffs) < min_matches:
        return None, len(diffs), True
    dd = pd.DataFrame(diffs, columns=["drone", "dz"])
    cnt = dd["drone"].value_counts()
    per = dd[dd["drone"].isin(cnt[cnt >= 5].index)].groupby("drone")["dz"].median()
    if per.empty:
        return None, len(diffs), True
    consistent = (per.max() - per.min()) <= 25
    return float(per.median()), len(diffs), bool(consistent)


def load_dataset(data_dir: str) -> Dataset:
    sensor_files = sorted(
        glob.glob(os.path.join(data_dir, "Delta_*.csv"))
        + glob.glob(os.path.join(data_dir, "Charlie_*.csv"))
    )
    drone_files = sorted(glob.glob(os.path.join(data_dir, "DroneTag_*.csv")))
    if not sensor_files and not drone_files:
        raise FileNotFoundError(f"No se han encontrado CSV en {data_dir}")

    sensors = [_read_sensor(f) for f in sensor_files]
    drones = [_read_drone(f) for f in drone_files]

    # Origen de la proyección: centro de las trayectorias + sensores
    lons = [dr.df["longitude"] for dr in drones] + [s.df["s_lon"] for s in sensors]
    lats = [dr.df["latitude"] for dr in drones] + [s.df["s_lat"] for s in sensors]
    lon0 = float(pd.concat(lons).median())
    lat0 = float(pd.concat(lats).median())
    proj = Projection(lon0, lat0)

    # Cota del terreno (altitud elipsoidal) estimada con los drones en tierra
    grounded = pd.concat([dr.df.loc[dr.df["height"].abs() < 1.0, "geo_altitude"] for dr in drones]) \
        if drones else pd.Series(dtype=float)
    ground_ref = float(grounded.median()) if len(grounded) else 0.0

    for dr in drones:
        dr.df["e"], dr.df["n"] = proj.enu(dr.df["longitude"], dr.df["latitude"])
        dr.df["z_raw"] = dr.df["geo_altitude"]
        dr.df["z_agl"] = dr.df["geo_altitude"] - ground_ref
    drones_d = {dr.key: dr for dr in drones}

    for s in sensors:
        d = s.df
        s.pos_e, s.pos_n = map(float, proj.enu(d["s_lon"].median(), d["s_lat"].median()))
        s.pos_z_raw = float(d["s_z"].median())
        # Si la cota del sensor es muy inferior al terreno, se asume altura sobre el suelo
        sensor_is_agl = s.pos_z_raw < ground_ref - 20
        s.pos_z_agl = s.pos_z_raw if sensor_is_agl else s.pos_z_raw - ground_ref

        if not s.bearing_only:
            d["e"], d["n"] = proj.enu(d["lon"], d["lat"])
            off, nmatch, consistent = _estimate_z_offset(s, drones_d)
            if off is not None and not consistent:
                s.z_offset = 0.0
                s.z_note = (f"desfase vertical no homogéneo entre drones ({nmatch} emparej.); "
                            "se mantienen las cotas originales")
            elif off is not None and abs(off) > 15:
                s.z_offset = off
                s.z_note = f"offset vertical estimado {off:+.1f} m ({nmatch} emparejamientos con DroneTag)"
            elif off is not None:
                s.z_offset = 0.0
                s.z_note = f"cotas coherentes con GNSS (desfase {off:+.1f} m, {nmatch} emparej.)"
            else:
                s.z_offset = ground_ref if sensor_is_agl else 0.0
                s.z_note = ("sin emparejamientos suficientes; se asume la misma referencia que el sensor"
                            f" (offset {s.z_offset:+.1f} m)")
            d["z_agl"] = d["z_raw"] + s.z_offset - ground_ref
        else:
            s.z_note = "sensor de sólo marcación (sin distancia): se dibujan rayos"

    t_all = [s.df["t"] for s in sensors] + [dr.df["t"] for dr in drones]
    t_all = pd.concat(t_all)
    return Dataset(sensors={s.key: s for s in sensors}, drones=drones_d, proj=proj,
                   ground_ref=ground_ref, t0=t_all.min().floor("s"), t1=t_all.max().ceil("s"))


def bearing_rays(sensor: Sensor, df: pd.DataFrame, length: float, use_agl: bool):
    """Calcula el extremo de cada rayo de marcación (azimut desde el norte, horario)."""
    az = np.radians(df["az"].values)
    el = np.radians(df["el"].values if "el" in df else np.zeros(len(df)))
    z0 = sensor.pos_z_agl if use_agl else sensor.pos_z_raw
    de = length * np.cos(el) * np.sin(az)
    dn = length * np.cos(el) * np.cos(az)
    du = length * np.sin(el)
    return sensor.pos_e + de, sensor.pos_n + dn, z0 + du
