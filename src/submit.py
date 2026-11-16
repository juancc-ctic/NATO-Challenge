"""Generación del Excel de entrega.

Convierte tracks internos (ENU por iteración) a WGS84 y covarianza NEU
centrada en cada state vector, y rellena la plantilla sin tocar los timestamps.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook

from src.geo import covariance_neu_to_neu, enu_to_geodetic
from src.parsing import parse_iso_to_ms
from src.schema import TEMPLATE_COLUMNS

_VALUE_COLUMNS = TEMPLATE_COLUMNS[2:]
_DEFAULTS: dict[str, Any] = {
    "classification": "Unknown",
    "classification_confidence": 0.0,
    "identity": "Unknown",
    "uav_model/manufacturer": "Unknown",
    "protocol": "Unknown",
    "serial_number": "Unknown",
    "signal_frequency_band": "Unknown",
}
_ALIASES: dict[str, tuple[str, ...]] = {
    "classification": ("class_norm",),
    "serial_number": ("serial_norm",),
}
_ALLOWED_SERIALS = (
    "F45TB21BR1AE03FP",
    "163DG1700154R3",
    "F5FHD233M00DEYG6",
    "1ZNBJAB00C00MQ",
    "1748FEV2HL7824051124",
    "Unknown",
    "Not Applicable",
)


def _serial_value(value: Any) -> str:
    """Ajusta el serial a los valores admitidos por la plantilla."""
    text = str(value).strip()
    if text in _ALLOWED_SERIALS:
        return text
    for allowed in _ALLOWED_SERIALS:
        if allowed in text:
            return allowed
    return "Unknown"


def _value(track: pd.Series, column: str) -> Any:
    """Devuelve el valor de una columna de track, con alias y default."""
    candidates = (column, *_ALIASES.get(column, ()))
    for name in candidates:
        if name not in track:
            continue
        value = track.get(name)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        if column == "serial_number":
            return _serial_value(value)
        return value
    return _DEFAULTS.get(column)


def _cell(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def tracks_to_submission(
    tracks: pd.DataFrame,
    origins: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Convierte tracks internos a filas WGS84 + covarianza NEU por state vector."""
    required = {
        "split_group",
        "ts_ms",
        "n",
        "e",
        "u",
        "C_nn",
        "C_ne",
        "C_nu",
        "C_ee",
        "C_eu",
        "C_uu",
        "track_id",
    }
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"Faltan columnas en tracks: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for _, track in tracks.iterrows():
        group = str(track["split_group"])
        if group not in origins:
            raise KeyError(f"No hay origen ENU para {group}")
        origin = origins[group]
        lat0 = float(origin["lat"])
        lon0 = float(origin["lon"])
        alt0 = float(origin["alt_m"])

        lat, lon, alt = enu_to_geodetic(
            float(track["e"]),
            float(track["n"]),
            float(track["u"]),
            lat0,
            lon0,
            alt0,
        )
        cov_in = np.array(
            [
                [float(track["C_nn"]), float(track["C_ne"]), float(track["C_nu"])],
                [float(track["C_ne"]), float(track["C_ee"]), float(track["C_eu"])],
                [float(track["C_nu"]), float(track["C_eu"]), float(track["C_uu"])],
            ],
            dtype=float,
        )
        cov_out = covariance_neu_to_neu(
            cov_in, lat0, lon0, float(lat), float(lon)
        )

        row: dict[str, Any] = {
            "ts_ms": int(track["ts_ms"]),
            "latitude": float(lat),
            "longitude": float(lon),
            "altitude_m": float(alt),
            "C_nn": float(cov_out[0, 0]),
            "C_ne": float(cov_out[0, 1]),
            "C_nu": float(cov_out[0, 2]),
            "C_ee": float(cov_out[1, 1]),
            "C_eu": float(cov_out[1, 2]),
            "C_uu": float(cov_out[2, 2]),
            "track_id": str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{group}:{track['track_id']}",
                )
            ),
        }
        for column in _VALUE_COLUMNS:
            if column not in row:
                row[column] = _value(track, column)
        rows.append(row)

    return pd.DataFrame(rows, columns=["ts_ms", *_VALUE_COLUMNS])


def write_submission(tracks: pd.DataFrame, template: Path, output: Path) -> None:
    """Escribe el Excel de entrega preservando `timestamp` y `unix_timestamp`."""
    if "ts_ms" in tracks.columns:
        timestamps = tracks["ts_ms"].astype("int64")
    elif "timestamp" in tracks.columns:
        timestamps = tracks["timestamp"].map(parse_iso_to_ms)
    else:
        raise ValueError("Los tracks deben tener `ts_ms` o `timestamp`")
    if timestamps.isna().any():
        raise ValueError("Hay tracks sin timestamp válido")

    by_timestamp = {
        int(key): value
        for key, value in tracks.assign(_ts=timestamps).groupby("_ts", sort=False)
    }

    source = load_workbook(template, read_only=True, data_only=True)
    sheet = source.active
    rows = sheet.iter_rows(values_only=True)
    try:
        header = next(rows)
    except StopIteration:
        raise ValueError(f"Plantilla vacía: {template}")
    if tuple(header) != tuple(TEMPLATE_COLUMNS):
        raise ValueError(f"Cabecera inesperada en {template}")

    output.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook(write_only=True)
    output_sheet = workbook.create_sheet(title=sheet.title)
    output_sheet.append(list(TEMPLATE_COLUMNS))

    for row in rows:
        timestamp = row[0] if len(row) > 0 else None
        unix_timestamp = row[1] if len(row) > 1 else None
        ts_ms = parse_iso_to_ms(timestamp)
        group = by_timestamp.get(ts_ms)
        if group is None or group.empty:
            output_sheet.append(
                [timestamp, unix_timestamp]
                + [None] * (len(TEMPLATE_COLUMNS) - 2)
            )
            continue
        for _, track in group.iterrows():
            values = [
                _cell(track.get(column, _DEFAULTS.get(column)))
                for column in _VALUE_COLUMNS
            ]
            output_sheet.append([timestamp, unix_timestamp] + values)

    workbook.save(output)
    source.close()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


_SIDECARS: dict[str, tuple[str, ...]] = {
    "classification": ("classification", "classification_confidence"),
    "identification": ("identity",),
    "metadata": (
        "uav_model/manufacturer",
        "protocol",
        "serial_number",
        "signal_frequency_band",
    ),
}
_SIDECAR_KEYS = ("split_group", "track_id")


def _load_sidecar(
    cfg: Any,
    name: str,
    columns: tuple[str, ...],
) -> pd.DataFrame | None:
    """Carga un sidecar opcional; devuelve `None` si no existe o está vacío."""
    path = cfg.reports / f"{name}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if frame.empty or not set(_SIDECAR_KEYS).issubset(frame.columns):
        return None
    keep = list(_SIDECAR_KEYS) + [column for column in columns if column in frame.columns]
    return frame[keep].drop_duplicates(subset=list(_SIDECAR_KEYS), keep="last")


def _merge_sidecars(
    tracks: pd.DataFrame,
    cfg: Any,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Mezcla los sidecars disponibles y deja `Unknown` lo que no exista."""
    merged = tracks.copy()
    status: dict[str, str] = {}
    for name, columns in _SIDECARS.items():
        sidecar = _load_sidecar(cfg, name, columns)
        if sidecar is None:
            status[name] = "not_implemented"
            continue
        present = [column for column in columns if column in sidecar.columns]
        status[name] = "used" if sidecar[present].notna().any().any() else "empty"
        merged = merged.drop(columns=present, errors="ignore")
        merged = merged.merge(
            sidecar,
            on=list(_SIDECAR_KEYS),
            how="left",
            validate="many_to_one",
        )
    return merged, status


def run_submission(
    tracks: pd.DataFrame,
    cfg: Any,
    split: str | None = None,
) -> dict[str, Any]:
    """Escribe un Excel de entrega por iteración a partir de los tracks reales."""
    from src.etl import discover_iterations

    origins_path = cfg.interim / "origins.json"
    origins = json.loads(origins_path.read_text(encoding="utf-8"))
    iterations = discover_iterations(cfg)
    merged, stages = _merge_sidecars(tracks, cfg)
    used_names: set[str] = set()
    results: list[dict[str, Any]] = []
    for iteration in iterations:
        if split is not None and iteration.split != split:
            continue
        if iteration.template_file is None:
            results.append(
                {
                    "split_group": iteration.split_group,
                    "template": None,
                    "output": None,
                    "status": "missing_template",
                }
            )
            continue
        group = iteration.split_group
        if group not in origins:
            raise KeyError(f"No hay origen ENU para {group}")
        subset = merged[merged["split_group"] == group] if not merged.empty else merged
        submission = tracks_to_submission(subset, origins)
        name = iteration.template_file.stem
        if name in used_names:
            name = f"{_safe_name(group)}_{name}"
        used_names.add(name)
        output = cfg.out / f"{name}.xlsx"
        write_submission(submission, iteration.template_file, output)
        results.append(
            {
                "split_group": group,
                "template": str(iteration.template_file),
                "output": str(output),
                "status": "ok",
                "rows": int(len(submission)),
            }
        )
    return {
        "n_iterations": len(results),
        "n_written": sum(1 for item in results if item["status"] == "ok"),
        "stages": stages,
        "results": results,
    }
