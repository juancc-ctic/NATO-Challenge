"""Capa del Excel de entrega: pinta exactamente lo que se va a enviar y lo comprueba.

Nos puntúan por el Excel (`out/<plantilla>.xlsx`, generado por `python -m src.cli submit`),
no por `tracks.parquet`. Esta capa lo lee tal cual y verifica, antes de entregar:
cabecera, rejilla de 100 ms, valores permitidos, covarianzas, que las posiciones caen
en la zona y que coinciden con los tracks del tracker (conversión ENU -> WGS84 correcta).
"""
from __future__ import annotations

import glob
import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import tracks_layer as TL  # añade la raíz del repo a sys.path
from src.schema import CLASS_VOCAB, IDENTITY_VOCAB, TEMPLATE_COLUMNS

SUB_COLOR = "#ff7ad9"
CACHE_DIR = Path(tempfile.gettempdir()) / "nato_viewer_cache"


@dataclass
class Submission:
    path: str
    rows: pd.DataFrame                          # filas con track, coords del visor, 1 punto/s por track
    checks: list = field(default_factory=list)  # [(estado, texto)] con estado "ok" | "warn" | "fail"
    n_tracks: int = 0
    n_rows: int = 0


def find_submission(iteration_dir: str, sub_dir: str | None) -> str | None:
    """El Excel se llama como la plantilla de la iteración (`*Template*/scenario_X_iteration_Y.xlsx`)."""
    if not iteration_dir or not sub_dir:
        return None
    templates = glob.glob(os.path.join(iteration_dir, "*Template*", "*.xlsx"))
    for t in templates:
        cand = os.path.join(sub_dir, os.path.basename(t))
        if os.path.exists(cand):
            return cand
    return None


def _read_excel_cached(path: str) -> pd.DataFrame:
    """Leer el Excel tarda ~13 s; se cachea en Parquet (se invalida si cambia el fichero)."""
    st = os.stat(path)
    key = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:16]
    cache = CACHE_DIR / f"{key}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_excel(path, dtype={"track_id": str})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.astype({c: str for c in df.columns if df[c].dtype == object}).to_parquet(cache, index=False)
    return df


def load_submission(path: str, ds, t0: pd.Timestamp, run=None) -> Submission:
    raw = _read_excel_cached(path)
    checks = []
    # 1. Cabecera
    cols = list(raw.columns)
    if cols == TEMPLATE_COLUMNS:
        checks.append(("ok", "Cabecera: las 19 columnas de la plantilla, en orden"))
    else:
        missing = [c for c in TEMPLATE_COLUMNS if c not in cols]
        checks.append(("fail", f"Cabecera distinta de la plantilla (faltan: {', '.join(missing) or 'ninguna'}; "
                               f"orden {'correcto' if not missing and cols[:19] == TEMPLATE_COLUMNS else 'distinto'})"))
    df = raw.copy()
    df["track_id"] = df["track_id"].replace({"nan": np.nan, "None": np.nan})
    ts = pd.to_numeric(df["unix_timestamp"], errors="coerce")
    df["ts_ms"] = (ts * 1000).round().astype("Int64")
    # 2. Rejilla de 100 ms
    grid = np.sort(df["ts_ms"].dropna().unique().astype(np.int64))
    steps = np.diff(grid) if len(grid) > 1 else np.array([100])
    off = int((steps != 100).sum())
    checks.append(("ok" if off == 0 else "fail",
                   f"Rejilla de 100 ms: {len(grid)} instantes, "
                   + ("sin saltos" if off == 0 else f"{off} saltos distintos de 100 ms")))
    rows = df[df["track_id"].notna()].copy()
    for c in ["latitude", "longitude", "altitude_m", "C_nn", "C_ne", "C_nu", "C_ee", "C_eu", "C_uu"]:
        rows[c] = pd.to_numeric(rows[c], errors="coerce")
    # 3. Posiciones completas
    nan_pos = int(rows[["latitude", "longitude"]].isna().any(axis=1).sum())
    checks.append(("ok" if nan_pos == 0 else "fail",
                   f"{len(rows):,}".replace(",", ".") + " filas con track, "
                   + ("todas con lat/lon" if nan_pos == 0 else f"{nan_pos} sin lat/lon")))
    # 4. Covarianzas
    bad_cov = int(((rows["C_nn"] < 0) | (rows["C_ee"] < 0) | (rows["C_uu"] < 0)).sum()
                  + rows[["C_nn", "C_ee", "C_uu"]].isna().any(axis=1).sum())
    checks.append(("ok" if bad_cov == 0 else "fail",
                   "Covarianzas: varianzas presentes y no negativas" if bad_cov == 0
                   else f"Covarianzas: {bad_cov} filas con varianza negativa o vacía"))
    # 5. Vocabulario
    bad_cls = sorted(set(rows["classification"].dropna().astype(str)) - set(CLASS_VOCAB))
    bad_id = sorted(set(rows["identity"].dropna().astype(str)) - set(IDENTITY_VOCAB))
    if not bad_cls and not bad_id:
        checks.append(("ok", "Clase e identidad con valores admitidos por la plantilla"))
    else:
        checks.append(("fail", f"Valores no admitidos: clase {bad_cls or '–'}, identidad {bad_id or '–'}"))
    # Cuánto está relleno (distinto de 'Unknown'): también puntúa en la evaluación.
    filled = {c: float((~rows[c].astype(str).isin(["Unknown", "nan", "None"])).mean() * 100)
              for c in ["classification", "identity"]}
    checks.append(("ok" if min(filled.values()) > 0 else "warn",
                   f"Rellenas (≠ Unknown): clase {filled['classification']:.0f}% de filas, "
                   f"identidad {filled['identity']:.0f}%"))
    # Coordenadas del visor, 1 punto/s por track para dibujar
    rows["sec"] = rows["ts_ms"] // 1000
    draw = rows.sort_values("ts_ms").drop_duplicates(["track_id", "sec"]).copy()
    draw["ve"], draw["vn"] = ds.proj.enu(draw["longitude"].to_numpy(), draw["latitude"].to_numpy())
    draw["z_raw"] = draw["altitude_m"]
    draw["z_agl"] = draw["altitude_m"] - ds.ground_ref
    draw["ts"] = (pd.to_datetime(draw["ts_ms"].astype("int64"), unit="ms") - t0).dt.total_seconds()
    # 6. Zona: a menos de 20 km del centro del escenario
    far = int((np.hypot(draw["ve"], draw["vn"]) > 20000).sum())
    checks.append(("ok" if far == 0 else "fail",
                   "Posiciones dentro de la zona (< 20 km)" if far == 0 else f"{far} puntos a más de 20 km"))
    # 7. Coincidencia con tracks.parquet (comprueba la conversión ENU -> WGS84 de submit)
    if run is not None and len(run.tracks):
        tr = run.tracks[["sec", "ve", "vn"]]
        m = draw[["sec", "ve", "vn"]].merge(tr, on="sec", suffixes=("", "_t"))
        if len(m):
            m["d"] = np.hypot(m.ve - m.ve_t, m.vn - m.vn_t)
            best = m.groupby(["sec", "ve", "vn"]).d.min()
            med, p99 = float(best.median()), float(best.quantile(0.99))
            checks.append(("ok" if p99 < 5 else "warn",
                           f"Coincide con tracks.parquet: desviación mediana {med:.2f} m, p99 {p99:.1f} m"))
        n_sub, n_tr = rows["track_id"].nunique(), run.tracks["track_id"].nunique()
        checks.append(("ok" if n_sub == n_tr else "warn", f"Tracks: {n_sub} en el Excel, {n_tr} en tracks.parquet"))
    return Submission(path=path, rows=draw, checks=checks, n_tracks=int(rows["track_id"].nunique()),
                      n_rows=int(len(raw)))


def submission_traces(go, sub: Submission, t_now: float, t_lo: float, zc: str, z_max: float):
    """El Excel de entrega como líneas finas rosas (una por track), con el hover de lo que se envía."""
    if sub is None or sub.rows.empty:
        return []
    d = sub.rows[(sub.rows.ts <= t_now) & (sub.rows.ts >= t_lo)]
    xs, ys, zs, txt = [], [], [], []
    for tid, g in d.groupby("track_id"):
        xs += list(g.ve) + [None]
        ys += list(g.vn) + [None]
        zs += list(g[zc].fillna(0).clip(upper=z_max)) + [None]
        info = (g["classification"].astype(str) + " · " + g["identity"].astype(str)).tolist()
        txt += [f"Excel · track {str(tid)[:8]}<br>{i}" for i in info] + [None]
    return [go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=SUB_COLOR, width=2),
                         opacity=0.9, name="Excel de entrega", text=txt,
                         hovertemplate="%{text}<extra></extra>")]


def find_iterations(root: str) -> list[tuple[str, str]]:
    """Iteraciones bajo `root` (carpetas con `SensorData/`), con su etiqueta `test:S1:I1`…"""
    out = []
    for sd in glob.glob(os.path.join(root, "**", "SensorData"), recursive=True):
        it = os.path.normpath(os.path.abspath(os.path.dirname(sd)))
        label = TL.infer_split_group(it) or os.path.relpath(it, root)
        out.append((label, it))
    order = lambda x: (0 if x[0].startswith("test") else 1, x[0])
    return sorted(out, key=order)


def data_root_from_iteration(iteration_dir: str) -> str | None:
    """Sube desde la iteración hasta la carpeta que contiene test/eval (o Testing/Evaluation datasets)."""
    p = Path(iteration_dir).resolve()
    for parent in p.parents:
        if parent.name.lower() in ("test", "eval", "testing datasets", "evaluation datasets"):
            return str(parent.parent)
    return None
