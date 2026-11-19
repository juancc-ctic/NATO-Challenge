"""Comparación de trayectorias estimadas (tracker) con trayectorias reales (GT).

Existe para ver *dónde* y *cuándo* falla el tracker, no solo cuánto: empareja
cada instante los tracks con los drones reales con el mismo criterio que GOSPA
(`src/evaluate.py`: Húngaro con corte `evaluation.gospa_c_m`) y deriva tablas
por instante, por dron y por iteración. Lo usan el visor 3D
(`visualization/app.py`) y el informe estático (`scripts/compare_report.py`).

Diferencia con `src/evaluate.py`: aquí se evalúa en la rejilla de 1 s del GT, así
que los instantes sin ningún track también cuentan como drones perdidos; el
evaluador solo mira los instantes donde hay algún track.

Solo lee; no toca el tracker ni la configuración.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.evaluate import gospa
from src.geo import enu_to_geodetic


def _evaluation_cfg(cfg: Any) -> Mapping[str, Any]:
    if hasattr(cfg, "data"):
        return cfg.data.get("evaluation", {}) or {}
    return cfg.get("evaluation", {}) if isinstance(cfg, Mapping) else {}


def load_origins(path: str | Path) -> dict[str, dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def add_geodetic(frame: pd.DataFrame, origins: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Añade `lat`, `lon`, `alt_m` a una tabla en ENU usando el origen de su iteración.

    Necesario para pintar tracks junto a datos que usan otro origen (el visor).
    """
    out = frame.copy()
    out["lat"] = np.nan
    out["lon"] = np.nan
    out["alt_m"] = np.nan
    for group, part in out.groupby("split_group"):
        origin = origins.get(str(group))
        if origin is None:
            continue
        up = part["u"].fillna(0.0).to_numpy(dtype=float)
        lat, lon, alt = enu_to_geodetic(part["e"].to_numpy(dtype=float), part["n"].to_numpy(dtype=float), up,
                                        float(origin["lat"]), float(origin["lon"]), float(origin["alt_m"]))
        out.loc[part.index, "lat"] = lat
        out.loc[part.index, "lon"] = lon
        out.loc[part.index, "alt_m"] = np.where(part["u"].notna(), alt, np.nan)
    return out


def _positions(frame: pd.DataFrame) -> np.ndarray:
    # Mismo convenio que src/evaluate.py (NaN -> 0) para que los números coincidan.
    return np.nan_to_num(frame[["n", "e", "u"]].to_numpy(dtype=float), nan=0.0)


def match_tracks(
    tracks: pd.DataFrame, ground_truth: pd.DataFrame, cfg: Any = None, step_ms: int = 1000
) -> dict[str, pd.DataFrame]:
    """Empareja tracks y GT instante a instante.

    Devuelve tablas:
    - `pairs`: (split_group, ts_ms, drone_key, track_id, dist_m)
    - `missed`: drones GT sin track en ese instante
    - `false`: tracks sin dron GT en ese instante
    - `instants`: GOSPA y su descomposición por instante
    """
    ecfg = _evaluation_cfg(cfg)
    c = float(ecfg.get("gospa_c_m", 50.0))
    p = float(ecfg.get("gospa_p", 2.0))
    pairs, missed, false, instants = [], [], [], []
    for group, gt_group in ground_truth.groupby("split_group"):
        tr_group = tracks[tracks.split_group == group]
        gt_by = {k: g.sort_values("ts_ms") for k, g in gt_group.groupby("drone_key")}
        t0 = int(gt_group.ts_ms.min()) // step_ms * step_ms
        t1 = int(gt_group.ts_ms.max())
        tr_times = np.sort(tr_group.ts_ms.unique()) if len(tr_group) else np.array([], dtype=np.int64)
        tr_by_time = {t: g for t, g in tr_group.groupby("ts_ms")} if len(tr_group) else {}
        for ts in range(t0, t1 + 1, step_ms):
            keys, pos = [], []
            for key, g in gt_by.items():
                times = g.ts_ms.to_numpy(dtype=np.int64)
                if ts < times[0] or ts > times[-1]:
                    continue
                keys.append(key)
                pos.append([np.interp(ts, times, g[a].to_numpy(dtype=float)) for a in ("n", "e", "u")])
            gt_pos = np.nan_to_num(np.array(pos, dtype=float).reshape(-1, 3), nan=0.0)
            # Snapshot de tracks más cercano (rejilla de 100 ms), a ±step/2.
            snap = None
            if len(tr_times):
                i = int(np.clip(np.searchsorted(tr_times, ts), 1, len(tr_times) - 1)) if len(tr_times) > 1 else 0
                cand = tr_times[[max(i - 1, 0), i]]
                best = int(cand[np.argmin(np.abs(cand - ts))])
                if abs(best - ts) <= step_ms // 2:
                    snap = tr_by_time[best]
            tr_ids = snap.track_id.astype(str).tolist() if snap is not None else []
            tr_pos = _positions(snap) if snap is not None else np.empty((0, 3))
            if not keys and not tr_ids:
                continue
            value, matched = gospa(tr_pos, gt_pos, c=c, p=p)
            used_t = {r for r, _, _ in matched}
            used_g = {g for _, g, _ in matched}
            for r, g, d in matched:
                pairs.append((group, ts, keys[g], tr_ids[r], d))
            for g, key in enumerate(keys):
                if g not in used_g:
                    missed.append((group, ts, key))
            for r, tid in enumerate(tr_ids):
                if r not in used_t:
                    false.append((group, ts, tid))
            loc = sum(d**p for _, _, d in matched)
            instants.append((group, ts, value, loc, len(keys) - len(matched), len(tr_ids) - len(matched),
                             len(keys), len(tr_ids)))
    return {
        "pairs": pd.DataFrame(pairs, columns=["split_group", "ts_ms", "drone_key", "track_id", "dist_m"]),
        "missed": pd.DataFrame(missed, columns=["split_group", "ts_ms", "drone_key"]),
        "false": pd.DataFrame(false, columns=["split_group", "ts_ms", "track_id"]),
        "instants": pd.DataFrame(instants, columns=["split_group", "ts_ms", "gospa", "loc_sum", "n_missed", "n_false",
                                                    "n_gt", "n_tracks"]),
        "c": pd.DataFrame([{"c": c, "p": p}]),
    }


def drone_summary(match: Mapping[str, pd.DataFrame], ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Por dron GT: % del tiempo seguido, error, tracks usados, cambios de track y retraso."""
    pairs, missed = match["pairs"], match["missed"]
    rows = []
    for key, g in ground_truth.groupby("drone_key"):
        p = pairs[pairs.drone_key == key].sort_values("ts_ms")
        m = missed[missed.drone_key == key]
        n_total = len(p) + len(m)
        airborne = g[(g.height_m > 5) | (g.speed_mps > 2)]
        t_air = int(airborne.ts_ms.min()) if len(airborne) else None
        first = int(p.ts_ms.min()) if len(p) else None
        switches = int((p.track_id != p.track_id.shift()).sum() - 1) if len(p) else 0
        rows.append({
            "split_group": g.split_group.iloc[0],
            "drone_key": key,
            "drone": str(key).split(":")[-1].replace("DroneTag_", ""),
            "color": g.color.iloc[0],
            "model": g.model_norm.iloc[0],
            "seconds": n_total,
            "pct_tracked": 100.0 * len(p) / n_total if n_total else 0.0,
            "err_mean_m": float(p.dist_m.mean()) if len(p) else np.nan,
            "err_p90_m": float(p.dist_m.quantile(0.9)) if len(p) else np.nan,
            "n_tracks": int(p.track_id.nunique()),
            "id_switches": max(switches, 0),
            "first_track_delay_s": (first - t_air) / 1000 if (first is not None and t_air is not None) else np.nan,
        })
    return pd.DataFrame(rows)


def iteration_summary(match: Mapping[str, pd.DataFrame], drones: pd.DataFrame) -> pd.DataFrame:
    """Por iteración: GOSPA medio y su descomposición, tracks falsos y cambios de track."""
    inst = match["instants"]
    c, p = float(match["c"].c.iloc[0]), float(match["c"].p.iloc[0])
    false, pairs = match["false"], match["pairs"]
    rows = []
    for group, x in inst.groupby("split_group"):
        n = len(x)
        never = set(false[false.split_group == group].track_id) - set(pairs[pairs.split_group == group].track_id)
        d = drones[drones.split_group == group]
        rows.append({
            "split_group": group,
            "gospa_mean": float(x.gospa.mean()),
            # Contribución media de cada término a GOSPA^p, en metros (raíz p de la media).
            "gospa_loc": float((x.loc_sum.mean()) ** (1 / p)),
            "gospa_missed": float((x.n_missed.mean() * c**p / p) ** (1 / p)),
            "gospa_false": float((x.n_false.mean() * c**p / p) ** (1 / p)),
            "missed_per_s": float(x.n_missed.mean()),
            "false_per_s": float(x.n_false.mean()),
            "pct_tracked": float(d.pct_tracked.mean()) if len(d) else np.nan,
            "err_mean_m": float(d.err_mean_m.mean()) if len(d) else np.nan,
            "id_switches": int(d.id_switches.sum()) if len(d) else 0,
            "tracks_never_matched": len(never),
            "seconds": n,
        })
    return pd.DataFrame(rows)


def compare_runs(a: pd.DataFrame, b: pd.DataFrame, key: str, label_a: str = "A", label_b: str = "B") -> pd.DataFrame:
    """Une dos resúmenes (antes/después) y calcula la diferencia de cada métrica."""
    metrics = [c for c in a.columns if c not in (key, "split_group", "drone", "drone_key", "color", "model")
               and pd.api.types.is_numeric_dtype(a[c])]
    keys = [key] if key == "split_group" else ["split_group", key]
    m = a.merge(b, on=keys, how="outer", suffixes=(f"_{label_a}", f"_{label_b}"))
    for metric in metrics:
        m[f"{metric}_diff"] = m[f"{metric}_{label_b}"] - m[f"{metric}_{label_a}"]
    return m


def horizontal_ellipse(c_nn: float, c_ne: float, c_ee: float, k: float = 2.0, n_points: int = 36) -> np.ndarray:
    """Puntos (dn, de) de la elipse de incertidumbre horizontal a k sigmas."""
    cov = np.array([[c_nn, c_ne], [c_ne, c_ee]], dtype=float)
    if not np.all(np.isfinite(cov)):
        return np.empty((0, 2))
    values, vectors = np.linalg.eigh(cov)
    values = np.clip(values, 0.0, None)
    angle = np.linspace(0, 2 * np.pi, n_points)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    return (vectors @ (k * np.sqrt(values)[:, None] * circle)).T


def run_compare(tracks: pd.DataFrame, ground_truth: pd.DataFrame, cfg: Any = None) -> dict[str, pd.DataFrame]:
    """Atajo: emparejamiento + resúmenes por dron e iteración."""
    match = match_tracks(tracks, ground_truth, cfg)
    drones = drone_summary(match, ground_truth)
    match["drones"] = drones
    match["iterations"] = iteration_summary(match, drones)
    return match
