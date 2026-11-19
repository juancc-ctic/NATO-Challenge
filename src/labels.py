"""Preasignación de etiquetas: cada detección de test se empareja con el dron GT.

Existe porque las detecciones no dicen a qué dron real corresponden; sin ese
enlace no hay etiquetas para entrenar clasificación/identidad ni forma de
medir sensores. Solo usa `test` (con GT); `eval` queda como `no_gt`.

Flujo por iteración:
1. Interpola la posición GT de cada dron al instante de cada detección.
2. Error por detección: distancia horizontal (sensores con posición) o error de
   azimut (sensores solo-rumbo). Puntuación = error / puerta del tipo de sensor.
3. Etiqueta por detección: dron con menor puntuación si es <= 1; si otro dron
   también entra en la puerta, se marca `ambiguous`.
4. Voto por track crudo (`object_id`) y por serial; la etiqueta final prioriza
   serial > track > detección cuando el voto es suficientemente puro.
5. Informes de verificación en `reports/labels/`.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.config import Config
from src.geo import angle_difference_deg
from src.parsing import json_safe

LABEL_COLUMNS = [
    "row_uid",
    "split_group",
    "sensor_id",
    "sensor_type",
    "ts_ms",
    "label_mode",
    "label_status",
    "label_source",
    "label_drone_key",
    "label_det_drone_key",
    "label_err",
    "label_score",
    "label_color",
    "label_is_blue",
    "label_model",
    "label_model_type",
    "label_gt_static",
]


def _labels_cfg(cfg: Config) -> Mapping[str, Any]:
    return cfg.data.get("labels", {}) or {}


def model_type(model: Any, mapping: Mapping[str, str]) -> str:
    """Devuelve el tipo (ala fija, multirrotor, FPV) según el nombre del modelo GT."""
    text = str(model or "").lower()
    for token, kind in mapping.items():
        if token.lower() in text:
            return kind
    return "Unknown"


def drone_table(ground_truth: pd.DataFrame, lcfg: Mapping[str, Any]) -> pd.DataFrame:
    """Resumen por dron GT, con la bandera `gt_static` (la etiqueta no despega)."""
    static = lcfg.get("static_gt", {})
    max_h = float(static.get("max_height_m", 10.0))
    max_extent = float(static.get("max_extent_m", 60.0))
    rows = []
    for key, part in ground_truth.groupby("drone_key"):
        extent = float(np.hypot(part.n - part.n.iloc[0], part.e - part.e.iloc[0]).max())
        h_max = float(part.height_m.max()) if part.height_m.notna().any() else np.nan
        first = part.iloc[0]
        rows.append(
            {
                "drone_key": key,
                "split_group": first.split_group,
                "color": first.color,
                "is_blue": bool(first.is_blue),
                "model": first.model_norm,
                "model_type": model_type(first.model_norm, lcfg.get("model_types", {})),
                "t_start_ms": int(part.ts_ms.min()),
                "t_end_ms": int(part.ts_ms.max()),
                "height_max_m": h_max,
                "extent_m": extent,
                "gt_static": bool((h_max < max_h) and (extent < max_extent)),
            }
        )
    return pd.DataFrame(rows)


def _sensor_mode(part: pd.DataFrame, lcfg: Mapping[str, Any]) -> tuple[str | None, pd.Series]:
    """Decide si un sensor se compara por posición o por azimut y qué filas valen."""
    stype = str(part.sensor_type.iloc[0])
    gate_m = lcfg.get("gate_m", {})
    gate_deg = lcfg.get("gate_deg", {})
    pos_ok = part["pn"].notna() & part["pe"].notna() & (np.hypot(part["pn"], part["pe"]) < float(lcfg.get("max_range_m", 20000)))
    brg_ok = part.az_deg.notna() & part.sensor_n.notna() & part.has_sensor_location.fillna(False).astype(bool)
    if stype in gate_m and pos_ok.mean() > 0.5:
        return "pos", pos_ok
    if stype in gate_deg and brg_ok.mean() > 0.5:
        return "brg", brg_ok
    return None, pos_ok & False


def _errors(
    part: pd.DataFrame,
    mode: str,
    gt: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    keys: list[str],
    tol_ms: int,
    offset_ms: int = 0,
) -> np.ndarray:
    """Matriz (detecciones × drones) de error; `inf` fuera del intervalo del vuelo."""
    t = part.ts_ms.to_numpy(dtype=np.int64) + offset_ms
    out = np.full((len(part), len(keys)), np.inf)
    for j, key in enumerate(keys):
        times, n_gt, e_gt = gt[key]
        inside = (t >= times[0] - tol_ms) & (t <= times[-1] + tol_ms)
        if not inside.any():
            continue
        n_i = np.interp(t, times, n_gt)
        e_i = np.interp(t, times, e_gt)
        if mode == "pos":
            err = np.hypot(part["pn"].to_numpy() - n_i, part["pe"].to_numpy() - e_i)
        else:
            expected = np.degrees(np.arctan2(e_i - part.sensor_e.to_numpy(), n_i - part.sensor_n.to_numpy()))
            err = np.abs(angle_difference_deg(part.az_deg.to_numpy(), expected))
        out[inside, j] = err[inside]
    return out


def _offset_scan(part, mode, gt, keys, tol_ms, gate, offsets_s) -> tuple[float, float, float]:
    """Desfase temporal que minimiza el error mediano (latencia o reloj desplazado).

    Devuelve (mejor desfase, error mediano con ese desfase, error mediano sin desfase),
    medidos igual: mediana de las detecciones a menos de 3 puertas de algún dron.
    """
    medians: dict[float, float] = {}
    for offset in offsets_s:
        err = _errors(part, mode, gt, keys, tol_ms, int(round(offset * 1000))).min(axis=1)
        err = err[np.isfinite(err) & (err < 3 * gate)]
        if len(err) >= 20:
            medians[round(float(offset), 2)] = float(np.median(err))
    if not medians:
        return 0.0, np.nan, np.nan
    best = min(medians, key=medians.get)
    return best, medians[best], medians.get(0.0, np.nan)


def _vote(frame: pd.DataFrame, by: list[str], min_hits: int, min_purity: float) -> pd.DataFrame:
    """Voto mayoritario de `label_det_drone_key` dentro de cada grupo."""
    matched = frame[frame.label_status == "matched"]
    if matched.empty:
        return pd.DataFrame(columns=by + ["vote_key", "n_matched", "purity"])
    counts = matched.groupby(by + ["label_det_drone_key"]).size().rename("n").reset_index()
    total = counts.groupby(by).n.transform("sum")
    counts["purity"] = counts.n / total
    counts["n_matched"] = total
    top = counts.sort_values("n", ascending=False).drop_duplicates(by)
    top = top.rename(columns={"label_det_drone_key": "vote_key"})
    ok = (top.n_matched >= min_hits) & (top.purity >= min_purity)
    top.loc[~ok, "vote_key"] = None
    return top[by + ["vote_key", "n_matched", "purity"]]


def label_detections(
    detections: pd.DataFrame, ground_truth: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Etiqueta cada detección y devuelve (etiquetas, tablas de verificación, resumen)."""
    lcfg = _labels_cfg(cfg)
    tol_ms = int(float(lcfg.get("time_tolerance_s", 1.0)) * 1000)
    gate_m = {k: float(v) for k, v in lcfg.get("gate_m", {}).items()}
    gate_deg = {k: float(v) for k, v in lcfg.get("gate_deg", {}).items()}
    offsets_s = np.arange(-float(lcfg.get("offset_scan_s", 5.0)), float(lcfg.get("offset_scan_s", 5.0)) + 1e-9, 0.5)

    det = detections.copy()
    det["pn"] = det.n.where(det.has_position, det.raw_n)
    det["pe"] = det.e.where(det.has_position, det.raw_e)
    det["track_ref"] = det.object_id.fillna(det.track_id_raw).astype("string")
    drones = drone_table(ground_truth, lcfg)
    dinfo = drones.set_index("drone_key")

    det["label_mode"] = None
    det["label_status"] = "no_gt"
    det["label_det_drone_key"] = None
    det["label_err"] = np.nan
    det["label_score"] = np.nan
    sensor_rows: list[dict[str, Any]] = []

    for group, gt_part in ground_truth.groupby("split_group"):
        gt = {}
        for key, part in gt_part.groupby("drone_key"):
            part = part.sort_values("ts_ms")
            gt[key] = (part.ts_ms.to_numpy(dtype=np.int64), part.n.to_numpy(), part.e.to_numpy())
        keys = list(gt)
        in_group = det.split_group == group
        for sensor_id, part in det[in_group].groupby("sensor_id"):
            stype = str(part.sensor_type.iloc[0])
            mode, usable = _sensor_mode(part, lcfg)
            det.loc[part.index, "label_status"] = "unusable"
            if mode is None:
                sensor_rows.append({"split_group": group, "sensor_id": sensor_id, "sensor_type": stype, "mode": None, "n": len(part)})
                continue
            if mode == "pos":
                # Posición presente pero fuera de rango (ecos a ±100–200 km): es clutter, no un dato inutilizable.
                far = ~usable & part["pn"].notna() & part["pe"].notna()
                det.loc[far[far].index, "label_status"] = "clutter"
            part = part[usable]
            gate = gate_m[stype] if mode == "pos" else gate_deg[stype]
            err = _errors(part, mode, gt, keys, tol_ms)
            order = np.argsort(err, axis=1)
            best = err[np.arange(len(part)), order[:, 0]]
            second = err[np.arange(len(part)), order[:, 1]] if len(keys) > 1 else np.full(len(part), np.inf)
            matched = best <= gate
            ambiguous = matched & (second <= gate)
            status = np.where(ambiguous, "ambiguous", np.where(matched, "matched", "clutter"))
            status = np.where(np.isfinite(best), status, "no_gt")
            det.loc[part.index, "label_mode"] = mode
            det.loc[part.index, "label_status"] = status
            det.loc[part.index, "label_err"] = np.where(np.isfinite(best), best, np.nan)
            det.loc[part.index, "label_score"] = np.where(np.isfinite(best), best / gate, np.nan)
            det.loc[part.index, "label_det_drone_key"] = np.where(matched, np.array(keys, dtype=object)[order[:, 0]], None)
            offset, offset_err, zero_err = _offset_scan(part, mode, gt, keys, tol_ms, gate, offsets_s)
            ok = matched & ~ambiguous
            sensor_rows.append(
                {
                    "split_group": group,
                    "sensor_id": sensor_id,
                    "sensor_type": stype,
                    "mode": mode,
                    "unit": "m" if mode == "pos" else "deg",
                    "gate": gate,
                    "n": int(len(part)),
                    "pct_matched": float(ok.mean() * 100),
                    "pct_ambiguous": float(ambiguous.mean() * 100),
                    "pct_clutter": float((status == "clutter").mean() * 100),
                    "err_median": float(np.median(best[ok])) if ok.any() else np.nan,
                    "err_p90": float(np.quantile(best[ok], 0.9)) if ok.any() else np.nan,
                    "best_offset_s": offset,
                    "scan_err_at_0": zero_err,
                    "scan_err_at_best": offset_err,
                }
            )

    min_hits = int(lcfg.get("vote_min_hits", 5))
    min_purity = float(lcfg.get("vote_min_purity", 0.7))
    tracks = _vote(det[det.track_ref.notna()], ["split_group", "sensor_id", "track_ref"], min_hits, min_purity)
    serials = _vote(det[det.serial_norm.notna()], ["split_group", "serial_norm"], min_hits, min_purity)

    det = det.merge(tracks[["split_group", "sensor_id", "track_ref", "vote_key"]].rename(columns={"vote_key": "track_key"}),
                    on=["split_group", "sensor_id", "track_ref"], how="left")
    det = det.merge(serials[["split_group", "serial_norm", "vote_key"]].rename(columns={"vote_key": "serial_key"}),
                    on=["split_group", "serial_norm"], how="left")

    # Prioridad: serial > track > detección. Solo se propaga dentro de la puerta
    # (matched/ambiguous) para no etiquetar clutter lejano con el voto del track.
    near = det.label_status.isin(["matched", "ambiguous"])
    det["label_drone_key"] = None
    det["label_source"] = "none"
    for column, source in (("label_det_drone_key", "detection"), ("track_key", "track"), ("serial_key", "serial")):
        use = near & det[column].notna()
        det.loc[use, "label_drone_key"] = det.loc[use, column]
        det.loc[use, "label_source"] = source
    det.loc[det.label_source.isin(["track", "serial"]) & (det.label_status == "ambiguous"), "label_status"] = "matched"

    for column, field in (("label_color", "color"), ("label_is_blue", "is_blue"), ("label_model", "model"),
                          ("label_model_type", "model_type"), ("label_gt_static", "gt_static")):
        det[column] = det.label_drone_key.map(dinfo[field])

    labels = det[LABEL_COLUMNS].copy()
    tables = _verification(det, drones, pd.DataFrame(sensor_rows), tracks, serials)
    summary = _summary(det, drones, tables)
    return labels, tables, summary


def _verification(det, drones, sensors, tracks, serials) -> dict[str, pd.DataFrame]:
    """Tablas para comprobar si la asignación tiene sentido."""
    test = det[det.split_group.isin(drones.split_group.unique())]
    matched = test[test.label_status == "matched"]
    by_drone = matched.groupby(["label_drone_key", "sensor_type"]).size().unstack(fill_value=0)
    drones_out = drones.merge(by_drone, left_on="drone_key", right_index=True, how="left")
    ser_models = test[test.serial_norm.notna()].groupby(["split_group", "serial_norm"]).model.agg(
        lambda s: ", ".join(sorted(set(s.dropna().astype(str))))).rename("models").reset_index()
    serials = serials.merge(ser_models, on=["split_group", "serial_norm"], how="left")
    classes = (matched.assign(class_norm=matched.get("class_norm", "Unknown"))
               .groupby(["sensor_type", "sensor_id", "label_model_type", "class_norm"]).size().rename("n").reset_index())
    sources = test.groupby(["sensor_type", "label_status", "label_source"]).size().rename("n").reset_index()
    return {"sensors": sensors, "tracks": tracks, "serials": serials, "drones": drones_out,
            "class_vs_gt": classes, "status": sources}


def _summary(det, drones, tables) -> dict[str, Any]:
    """Resumen y avisos automáticos (lo que no tiene sentido o hay que revisar)."""
    test = det[det.split_group.isin(drones.split_group.unique())]
    warnings: list[str] = []
    sensors = tables["sensors"].dropna(subset=["mode"])
    for row in sensors.itertuples():
        if row.n >= 50 and abs(row.best_offset_s) >= 1.0 and row.scan_err_at_best < 0.8 * row.scan_err_at_0:
            warnings.append(
                f"{row.split_group} {row.sensor_id}: con {row.best_offset_s:+.1f} s el error mediano baja de "
                f"{row.scan_err_at_0:.1f} a {row.scan_err_at_best:.1f} {row.unit} (latencia o reloj desplazado)."
            )
    fusion = test[(test.sensor_type == "fusion") & (test.label_status == "matched")]
    if len(fusion):
        blue = float(fusion.label_is_blue.astype(bool).mean())
        if blue > 0.9:
            warnings.append(f"Fusión: el {blue:.0%} de sus detecciones emparejadas son drones BLUE; posible fuga de etiqueta amigo/enemigo.")
    for row in drones[drones.gt_static].itertuples():
        n = int((test.label_drone_key == row.drone_key).sum())
        warnings.append(f"{row.drone_key}: GT estático (no despega) y {n} detecciones emparejadas; revisar si la etiqueta iba en el dron.")
    serials = tables["serials"]
    for row in serials.itertuples():
        if row.n_matched >= 5 and row.purity < 0.7:
            warnings.append(f"{row.split_group} serial {row.serial_norm}: solo {row.purity:.0%} de sus detecciones caen en el mismo dron; no se usa para etiquetar.")
    # Un mismo serial debería ser siempre el mismo modelo de dron, también entre iteraciones.
    voted = serials[serials.vote_key.notna()].copy()
    voted["gt_model"] = voted.vote_key.map(drones.set_index("drone_key").model)
    for serial, part in voted.groupby("serial_norm"):
        models = sorted(set(part.gt_model.dropna()))
        if len(models) > 1:
            where = ", ".join(f"{r.split_group}→{r.gt_model}" for r in part.itertuples())
            warnings.append(f"Serial {serial} se asigna a modelos distintos ({where}); revisar GT o el serial.")
    status = test.label_status.value_counts(normalize=True).mul(100).round(1).to_dict()
    source = test[test.label_drone_key.notna()].label_source.value_counts().to_dict()
    tr = tables["tracks"]
    return {
        "n_detections_test": int(len(test)),
        "status_pct": status,
        "label_source": source,
        "n_labelled": int(test.label_drone_key.notna().sum()),
        "track_purity_median": float(tr.purity.median()) if len(tr) else None,
        "tracks_pure_pct": float((tr.vote_key.notna()).mean() * 100) if len(tr) else None,
        "n_drones": int(len(drones)),
        "n_static_drones": int(drones.gt_static.sum()),
        "warnings": warnings,
    }


def run_labels(cfg: Config) -> dict[str, Any]:
    """Lee interim, etiqueta y escribe `detection_labels.parquet` e informes."""
    detections = pd.read_parquet(cfg.interim / "detections.parquet")
    ground_truth = pd.read_parquet(cfg.interim / "ground_truth.parquet")
    labels, tables, summary = label_detections(detections, ground_truth, cfg)
    output = cfg.interim / "detection_labels.parquet"
    labels.to_parquet(output, index=False)
    folder = cfg.reports / "labels"
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", index=False)
    with open(folder / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2, ensure_ascii=False)
    summary["outputs"] = {"labels": str(output), "reports": str(folder)}
    return summary
