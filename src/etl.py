"""ETL principal del challenge.

Existe porque cada sensor tiene un esquema distinto; por eso descubre los CSV,
normaliza todo a una tabla común, calcula ENU, deduplica y escribe `data/interim/`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from src.classify import flatten_classification, map_class
from src.config import Config
from src.geo import geodetic_to_enu, median_origin, normalize_azimuth, valid_latlon
from src.identify import load_friendly_serials, match_friendly, normalize_serial
from src.metadata import extract_metadata, lookup_value, meta_items, normalize_model, infer_manufacturer
from src.parsing import (
    clean_text,
    json_safe,
    ms_to_iso,
    parse_gt_filename,
    parse_iso_to_ms,
    parse_sensor_filename,
    read_csv_tolerant,
    safe_float,
    sha256_file,
)
from src.schema import (
    BEARING_ONLY,
    DETECTION_COLUMNS,
    FRIENDLY_COLUMNS,
    GT_COLUMNS,
    INVENTORY_COLUMNS,
    POSITION_CAPABLE,
    TEST_ONLY,
)

SENSOR_TYPE_MAP = {
    "RADAR": "radar",
    "PASSIVE-RF": "rf",
    "ACOUSTIC": "acoustic",
    "CAMERA": "camera",
    "CYBER": "cyber",
    "FUSION-NODE": "fusion",
    "OTHER": "other",
    "PASSIVE-RF--JAMMER--CYBER": "rf",
}


@dataclass(frozen=True)
class IterationPaths:
    split: str
    dataset: str
    scenario: int
    iteration: int
    split_group: str
    root: Path
    sensor_files: tuple[Path, ...]
    gt_files: tuple[Path, ...]
    friendly_files: tuple[Path, ...]
    template_file: Path | None


@dataclass
class EtlResult:
    detections: pd.DataFrame
    ground_truth: pd.DataFrame
    friendly: pd.DataFrame
    inventory: pd.DataFrame
    issues: pd.DataFrame
    origins: dict[str, dict[str, Any]]
    summary: dict[str, Any]


def _relative(path: Path, cfg: Config) -> str:
    for base in (cfg.root, cfg.root.parent):
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    return str(path)


def _number(value: str) -> int:
    match = re.search(r"(\d+)", value)
    return int(match.group(1)) if match else 0


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _find_dir(parent: Path, token: str) -> Path | None:
    if not parent.is_dir():
        return None
    for child in sorted(parent.iterdir()):
        if child.is_dir() and token in child.name.lower():
            return child
    return None


def _find_template(parent: Path) -> Path | None:
    if not parent.is_dir():
        return None
    for child in sorted(parent.iterdir()):
        if child.is_dir() and "template" in child.name.lower():
            for path in sorted(child.glob("*.xlsx")):
                return path
    return None


def discover_iterations(cfg: Config) -> list[IterationPaths]:
    """Devuelve las iteraciones y ficheros encontrados bajo `data/raw/`."""
    iterations: list[IterationPaths] = []
    roots: list[tuple[str, str, Path]] = [("test", "Testing", cfg.raw_test)]
    if cfg.raw_eval.is_dir():
        for dataset_dir in sorted(cfg.raw_eval.iterdir()):
            if dataset_dir.is_dir() and dataset_dir.name.lower().startswith("dataset"):
                roots.append(("eval", dataset_dir.name, dataset_dir))
    for split, dataset, root in roots:
        if not root.is_dir():
            continue
        for scenario_dir in sorted(root.iterdir()):
            if not scenario_dir.is_dir() or not scenario_dir.name.lower().startswith("scenario"):
                continue
            scenario = _number(scenario_dir.name)
            for iteration_dir in sorted(scenario_dir.iterdir()):
                if not iteration_dir.is_dir() or not iteration_dir.name.lower().startswith("iteration"):
                    continue
                iteration = _number(iteration_dir.name)
                sensor_dir = _find_dir(iteration_dir, "sensordata")
                gt_dir = _find_dir(iteration_dir, "truth")
                friendly_dir = _find_dir(iteration_dir, "friend")
                template_file = _find_template(iteration_dir)
                sensor_files = tuple(sorted(sensor_dir.glob("*.csv"))) if sensor_dir else ()
                gt_files = tuple(sorted(gt_dir.glob("*.csv"))) if gt_dir else ()
                friendly_files = tuple(
                    sorted(list(friendly_dir.glob("*.xlsx")) + list(friendly_dir.glob("*.csv")))
                ) if friendly_dir else ()
                if split == "test":
                    split_group = f"test:S{scenario}:I{iteration}"
                else:
                    split_group = f"eval:{_sanitize(dataset)}:S{scenario}:I{iteration}"
                iterations.append(
                    IterationPaths(
                        split=split,
                        dataset=dataset,
                        scenario=scenario,
                        iteration=iteration,
                        split_group=split_group,
                        root=iteration_dir,
                        sensor_files=sensor_files,
                        gt_files=gt_files,
                        friendly_files=friendly_files,
                        template_file=template_file,
                    )
                )
    return iterations


def _is_radians(coord_system: Any) -> bool:
    text = str(coord_system).upper() if coord_system is not None else ""
    return "RAD" in text or "RADIAN" in text


def _valid_coordinate(lat: float, lon: float) -> bool:
    return bool(valid_latlon(np.asarray([lat]), np.asarray([lon]))[0])


def _close_latlon(lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
    if not (math.isfinite(lat1) and math.isfinite(lon1) and math.isfinite(lat2) and math.isfinite(lon2)):
        return False
    dn = (lat1 - lat2) * 110540.0
    de = (lon1 - lon2) * 111320.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.hypot(dn, de) < 5.0


def _row_uid(split_group: str, sensor_id: str, row: dict[str, Any]) -> str:
    content = "|".join(f"{key}={row.get(key, '')}" for key in sorted(row))
    payload = f"{split_group}|{sensor_id}|{content}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _sigma(
    row: dict[str, Any],
    sensor_type: str,
    lat: float,
    cfg: Config,
) -> tuple[float, float, float, str]:
    sx = safe_float(row.get("location_xError"))
    sy = safe_float(row.get("location_yError"))
    sz = safe_float(row.get("location_zError"))
    defaults = cfg.etl.get("default_sigma_m", {})
    default = float(defaults.get(sensor_type, 50.0))
    floor = float(cfg.etl.get("sigma_floor_m", 0.5))
    cap = float(cfg.etl.get("sigma_cap_m", 5000.0))
    sigma_n = math.nan
    sigma_e = math.nan
    sigma_u = math.nan
    converted = False
    reported = False
    if math.isfinite(sy) and sy > 0:
        if sensor_type == "cyber" and sy < 0.01:
            sigma_n = sy * 110540.0
            converted = True
        else:
            sigma_n = sy
            reported = True
    if math.isfinite(sx) and sx > 0:
        if sensor_type == "cyber" and sx < 0.01:
            sigma_e = sx * 111320.0 * math.cos(math.radians(lat))
            converted = True
        else:
            sigma_e = sx
            reported = True
    if math.isfinite(sz) and sz > 0:
        sigma_u = sz
        reported = True
    used_default = False
    if not math.isfinite(sigma_n):
        sigma_n = default
        used_default = True
    if not math.isfinite(sigma_e):
        sigma_e = default
        used_default = True
    if not math.isfinite(sigma_u):
        sigma_u = default
        used_default = True
    clipped = False
    for name, value in (("n", sigma_n), ("e", sigma_e), ("u", sigma_u)):
        if value < floor or value > cap:
            clipped = True
        if name == "n":
            sigma_n = min(max(value, floor), cap)
        elif name == "e":
            sigma_e = min(max(value, floor), cap)
        else:
            sigma_u = min(max(value, floor), cap)
    if used_default:
        source = "default"
    elif clipped:
        source = "capped"
    elif converted:
        source = "reported_deg_converted"
    elif reported:
        source = "reported_m"
    else:
        source = "none"
    return sigma_n, sigma_e, sigma_u, source


def parse_sensor_file(
    path: Path,
    paths: IterationPaths,
    cfg: Config,
    friendly_serials: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Parsea un CSV de sensor a registros del esquema común."""
    header, rows, malformed = read_csv_tolerant(path)
    file_meta = parse_sensor_filename(path)
    platform = file_meta["platform"]
    sensor_code = file_meta["sensor_code"]
    sensor_type_raw = file_meta["sensor_type_raw"]
    sensor_type = SENSOR_TYPE_MAP.get(sensor_type_raw.upper(), "other")
    sensor_id = f"{platform}_{sensor_code}"
    bands = cfg.etl.get("frequency_bands_mhz", [])
    position_capable = set(cfg.etl.get("position_capable_sensor_types", POSITION_CAPABLE))
    bearing_only = set(cfg.etl.get("bearing_only_sensor_types", BEARING_ONLY))
    test_only = set(cfg.etl.get("test_only_sensor_types", TEST_ONLY))
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    missing_ts = 0
    missing_sensor_loc = 0
    with_location = 0
    with_range_bearing = 0
    node_ids: set[str] = set()
    coord_systems: set[str] = set()
    datums: set[str] = set()
    t_min: int | None = None
    t_max: int | None = None
    friendly_available = friendly_serials is not None

    for index, row in enumerate(rows):
        ts_ms = parse_iso_to_ms(row.get("timestamp"))
        if ts_ms is None:
            missing_ts += 1
            issues.append(
                {
                    "split_group": paths.split_group,
                    "source_file": path.name,
                    "source_relpath": _relative(path, cfg),
                    "raw_index": index,
                    "issue": "missing_or_invalid_timestamp",
                    "detail": clean_text(row.get("timestamp")),
                }
            )
            continue
        node_id = clean_text(row.get("nodeId"))
        if node_id:
            node_ids.add(node_id)
        sensor_cs = (
            clean_text(row.get("sensor_location_coordinateSystem"))
            or clean_text(row.get("location_coordinateSystem"))
            or clean_text(row.get("rangeBearing_coordinateSystem"))
        )
        location_cs = clean_text(row.get("location_coordinateSystem")) or sensor_cs
        datum = (
            clean_text(row.get("location_datum"))
            or clean_text(row.get("sensor_location_datum"))
            or clean_text(row.get("rangeBearing_datum"))
        )
        if sensor_cs:
            coord_systems.add(sensor_cs)
        if location_cs:
            coord_systems.add(location_cs)
        if datum:
            datums.add(datum)

        sensor_lon = safe_float(row.get("sensor_location_x"))
        sensor_lat = safe_float(row.get("sensor_location_y"))
        sensor_alt = safe_float(row.get("sensor_location_z"))
        if _is_radians(sensor_cs):
            sensor_lat = math.degrees(sensor_lat)
            sensor_lon = math.degrees(sensor_lon)
        has_sensor_location = _valid_coordinate(sensor_lat, sensor_lon) and not (
            abs(sensor_lat) < 1e-9 and abs(sensor_lon) < 1e-9
        )
        if not has_sensor_location:
            sensor_lat = math.nan
            sensor_lon = math.nan
            sensor_alt = math.nan
            missing_sensor_loc += 1
            issues.append(
                {
                    "split_group": paths.split_group,
                    "source_file": path.name,
                    "source_relpath": _relative(path, cfg),
                    "raw_index": index,
                    "issue": "missing_or_invalid_sensor_location",
                    "detail": "zero_placeholder_or_invalid",
                }
            )

        raw_lon = safe_float(row.get("location_x"))
        raw_lat = safe_float(row.get("location_y"))
        raw_alt = safe_float(row.get("location_z"))
        if _is_radians(location_cs):
            raw_lat = math.degrees(raw_lat)
            raw_lon = math.degrees(raw_lon)
        has_raw_location = _valid_coordinate(raw_lat, raw_lon) and not (
            abs(raw_lat) < 1e-9 and abs(raw_lon) < 1e-9
        )
        if has_raw_location:
            with_location += 1
        else:
            raw_lat = math.nan
            raw_lon = math.nan
            raw_alt = math.nan

        az = safe_float(row.get("rangeBearing_azimuth"))
        el = safe_float(row.get("rangeBearing_elevation"))
        range_m = safe_float(row.get("rangeBearing_range"))
        bearing_cs = clean_text(row.get("rangeBearing_coordinateSystem"))
        if _is_radians(bearing_cs):
            az = math.degrees(az)
            el = math.degrees(el)
        object_items = meta_items(row.get("objectInfo"))
        if not math.isfinite(az):
            az = safe_float(lookup_value(object_items, "az", "azimuth"))
        if not math.isfinite(el):
            el = safe_float(lookup_value(object_items, "el", "elevation"))
        if math.isfinite(az):
            az = float(normalize_azimuth(az))
        has_bearing = math.isfinite(az) or math.isfinite(el)
        if has_bearing:
            with_range_bearing += 1

        has_position = (
            sensor_type in position_capable
            and sensor_type not in bearing_only
            and has_raw_location
        )
        if has_position:
            lat = raw_lat
            lon = raw_lon
            alt_m = raw_alt
            pos_source = "location"
        else:
            lat = math.nan
            lon = math.nan
            alt_m = math.nan
            if has_raw_location and sensor_type not in position_capable:
                if _close_latlon(raw_lat, raw_lon, sensor_lat, sensor_lon):
                    pos_source = "sensor_location_placeholder"
                else:
                    pos_source = "raw_bearing_only"
            else:
                pos_source = "none"
        has_alt = has_position and math.isfinite(alt_m)

        sigma_n, sigma_e, sigma_u, sigma_source = _sigma(row, sensor_type, raw_lat, cfg)
        if not has_position:
            sigma_n = math.nan
            sigma_e = math.nan
            sigma_u = math.nan
            sigma_source = "none"

        class_raw, class_conf, class_path, class_top, class_top_conf = flatten_classification(
            row.get("classification")
        )
        class_norm, class_is_drone, class_is_non_drone, class_is_transmitter = map_class(class_raw, class_path, class_top)
        has_class = class_norm != "Unknown" and class_conf > 0.0

        meta = extract_metadata(row, bands)
        serial = meta["serial"]
        serial_norm = normalize_serial(serial)
        model = meta["model"]
        model_norm = meta["model_norm"]
        manufacturer = meta["manufacturer"]
        protocol = meta["protocol"]
        freq_mhz = meta["freq_mhz"]
        freq_band = meta["freq_band"]
        has_serial = serial_norm is not None
        has_metadata = any(
            value is not None
            for value in (serial, model, protocol, manufacturer, freq_band)
        ) or math.isfinite(freq_mhz)

        vn = safe_float(row.get("enuVelocity_northRate"))
        ve = safe_float(row.get("enuVelocity_eastRate"))
        vu = safe_float(row.get("enuVelocity_upRate"))
        sigma_vn = safe_float(row.get("enuVelocity_northRateError"))
        sigma_ve = safe_float(row.get("enuVelocity_eastRateError"))
        sigma_vu = safe_float(row.get("enuVelocity_upRateError"))
        has_velocity = any(
            math.isfinite(value) for value in (vn, ve, vu)
        )

        detection_confidence = safe_float(row.get("detectionConfidence"))
        if not math.isfinite(detection_confidence):
            detection_confidence = 1.0

        if t_min is None or ts_ms < t_min:
            t_min = ts_ms
        if t_max is None or ts_ms > t_max:
            t_max = ts_ms

        record = {
            "ts_ms": ts_ms,
            "ts_iso": ms_to_iso(ts_ms),
            "sensor_id": sensor_id,
            "sensor_type": sensor_type,
            "sensor_type_raw": sensor_type_raw,
            "sensor_code": sensor_code,
            "platform": platform,
            "sensor_node_id": node_id,
            "lat": lat,
            "lon": lon,
            "alt_m": alt_m,
            "n": math.nan,
            "e": math.nan,
            "u": math.nan,
            "sigma_n": sigma_n,
            "sigma_e": sigma_e,
            "sigma_u": sigma_u,
            "az_deg": az if math.isfinite(az) else math.nan,
            "el_deg": el if math.isfinite(el) else math.nan,
            "range_m": range_m,
            "has_position": has_position,
            "has_bearing": has_bearing,
            "has_alt": has_alt,
            "has_class": has_class,
            "has_serial": has_serial,
            "has_velocity": has_velocity,
            "has_metadata": has_metadata,
            "class_raw": class_raw,
            "class_conf": class_conf,
            "class_path": " > ".join(class_path),
            "class_top": class_top,
            "class_norm": class_norm,
            "class_is_drone": class_is_drone,
            "class_is_non_drone": class_is_non_drone,
            "class_is_transmitter": class_is_transmitter,
            "serial": serial,
            "serial_norm": serial_norm,
            "model": model,
            "model_norm": model_norm,
            "manufacturer": manufacturer,
            "protocol": protocol,
            "freq_mhz": freq_mhz,
            "freq_band": freq_band,
            "det_conf": detection_confidence,
            "vn": vn,
            "ve": ve,
            "vu": vu,
            "sigma_vn": sigma_vn,
            "sigma_ve": sigma_ve,
            "sigma_vu": sigma_vu,
            "drone_mac": meta["drone_mac"],
            "ap_mac": meta["ap_mac"],
            "pilot_mac": meta["pilot_mac"],
            "remote_id": meta["remote_id"],
            "heading_deg": meta["heading_deg"],
            "channel": meta["channel"],
            "event_id": meta["event_id"],
            "event_serial": meta["event_serial"],
            "event_factory": meta["event_factory"],
            "event_takeover": meta["event_takeover"],
            "object_id": clean_text(row.get("objectId")),
            "task_id": clean_text(row.get("taskId")),
            "report_id": clean_text(row.get("reportId")),
            "track_id_raw": meta["track_id_raw"],
            "colour": clean_text(row.get("colour")),
            "state": clean_text(row.get("state")),
            "coord_system_raw": location_cs,
            "datum_raw": datum,
            "sensor_lat": sensor_lat,
            "sensor_lon": sensor_lon,
            "sensor_alt_m": sensor_alt,
            "sensor_n": math.nan,
            "sensor_e": math.nan,
            "sensor_u": math.nan,
            "has_sensor_location": has_sensor_location,
            "raw_lat": raw_lat,
            "raw_lon": raw_lon,
            "raw_alt_m": raw_alt,
            "raw_n": math.nan,
            "raw_e": math.nan,
            "raw_u": math.nan,
            "pos_source": pos_source,
            "sigma_source": sigma_source,
            "alt_source": "reported" if has_alt else "missing",
            "split": paths.split,
            "dataset": paths.dataset,
            "scenario": paths.scenario,
            "iteration": paths.iteration,
            "split_group": paths.split_group,
            "split_role": "dev" if paths.split == "test" else "holdout",
            "source_file": path.name,
            "source_relpath": _relative(path, cfg),
            "raw_index": index,
            "row_uid": _row_uid(paths.split_group, sensor_id, row),
            "is_fusion_node": sensor_type == "fusion",
            "is_test_only": sensor_type in test_only,
            "gt_available": paths.split == "test",
            "friendly_available": friendly_available,
            "friendly_serial_match": match_friendly(serial, friendly_serials) if friendly_serials else False,
        }
        records.append(record)

    file_start_ms = file_meta["file_start_ms"]
    file_end_ms = file_meta["file_end_ms"]
    stat = {
        "split": paths.split,
        "dataset": paths.dataset,
        "scenario": paths.scenario,
        "iteration": paths.iteration,
        "split_group": paths.split_group,
        "source_kind": "sensor",
        "source_file": path.name,
        "source_relpath": _relative(path, cfg),
        "sensor_type_raw": sensor_type_raw,
        "sensor_type": sensor_type,
        "sensor_code": sensor_code,
        "platform": platform,
        "node_ids": "|".join(sorted(node_ids)),
        "n_rows_raw": len(rows) + malformed,
        "n_rows_parsed": len(records),
        "n_rows_malformed": malformed,
        "n_rows_missing_ts": missing_ts,
        "n_rows_missing_sensor_loc": missing_sensor_loc,
        "n_rows_with_location": with_location,
        "n_rows_with_range_bearing": with_range_bearing,
        "n_rows_out": len(records),
        "t_min_iso": ms_to_iso(t_min) if t_min is not None else ms_to_iso(file_start_ms),
        "t_max_iso": ms_to_iso(t_max) if t_max is not None else ms_to_iso(file_end_ms),
        "t_min_ms": t_min if t_min is not None else file_start_ms,
        "t_max_ms": t_max if t_max is not None else file_end_ms,
        "file_start_iso": ms_to_iso(file_start_ms),
        "file_end_iso": ms_to_iso(file_end_ms),
        "file_start_ms": file_start_ms,
        "file_end_ms": file_end_ms,
        "coord_systems": "|".join(sorted(coord_systems)),
        "datums": "|".join(sorted(datums)),
        "columns": "|".join(header),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "model_raw": None,
        "color": None,
        "n_serials": 0,
    }
    if malformed:
        issues.append(
            {
                "split_group": paths.split_group,
                "source_file": path.name,
                "source_relpath": _relative(path, cfg),
                "raw_index": None,
                "issue": "malformed_csv_row",
                "detail": f"{malformed} rows with field count mismatch",
            }
        )
    return records, stat, issues


def parse_gt_file(
    path: Path,
    paths: IterationPaths,
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Parsea un CSV Dronetag de ground truth."""
    header, rows, malformed = read_csv_tolerant(path)
    file_meta = parse_gt_filename(path)
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    missing_ts = 0
    t_min: int | None = None
    t_max: int | None = None
    model_raw = file_meta["model_raw"]
    model_norm = normalize_model(model_raw) or "Unknown"
    manufacturer = infer_manufacturer(model_raw)
    for index, row in enumerate(rows):
        ts_ms = parse_iso_to_ms(row.get("timestamp"))
        if ts_ms is None:
            missing_ts += 1
            issues.append(
                {
                    "split_group": paths.split_group,
                    "source_file": path.name,
                    "source_relpath": _relative(path, cfg),
                    "raw_index": index,
                    "issue": "missing_or_invalid_gt_timestamp",
                    "detail": clean_text(row.get("timestamp")),
                }
            )
            continue
        lat = safe_float(row.get("latitude"))
        lon = safe_float(row.get("longitude"))
        if not _valid_coordinate(lat, lon):
            issues.append(
                {
                    "split_group": paths.split_group,
                    "source_file": path.name,
                    "source_relpath": _relative(path, cfg),
                    "raw_index": index,
                    "issue": "missing_or_invalid_gt_location",
                    "detail": None,
                }
            )
            continue
        geo_alt = safe_float(row.get("geo_altitude"))
        altitude_raw = safe_float(row.get("altitude"))
        if math.isfinite(geo_alt):
            alt_m = geo_alt
            alt_source = "geo_altitude"
        elif math.isfinite(altitude_raw):
            alt_m = altitude_raw
            alt_source = "altitude_fallback"
        else:
            alt_m = math.nan
            alt_source = "missing"
        horizontal_accuracy = safe_float(row.get("horizontal_accuracy"))
        vertical_accuracy = safe_float(row.get("vertical_accuracy"))
        speed_accuracy = safe_float(row.get("speed_accuracy"))
        velocity_x = safe_float(row.get("velocity_x"))
        velocity_y = safe_float(row.get("velocity_y"))
        velocity_z = safe_float(row.get("velocity_z"))
        if all(math.isfinite(value) for value in (velocity_x, velocity_y, velocity_z)):
            speed_mps = math.sqrt(velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2)
        else:
            speed_mps = math.nan
        if t_min is None or ts_ms < t_min:
            t_min = ts_ms
        if t_max is None or ts_ms > t_max:
            t_max = ts_ms
        records.append(
            {
                "ts_ms": ts_ms,
                "ts_iso": ms_to_iso(ts_ms),
                "drone_key": f"{paths.split_group}:{path.stem}",
                "color": file_meta["color"],
                "is_blue": file_meta["color"] == "BLUE",
                "model_raw": model_raw,
                "model_norm": model_norm,
                "manufacturer": manufacturer,
                "lat": lat,
                "lon": lon,
                "alt_m": alt_m,
                "altitude_raw": altitude_raw,
                "height_m": safe_float(row.get("height")),
                "n": math.nan,
                "e": math.nan,
                "u": math.nan,
                "sigma_n": horizontal_accuracy,
                "sigma_e": horizontal_accuracy,
                "sigma_u": vertical_accuracy,
                "velocity_x": velocity_x,
                "velocity_y": velocity_y,
                "velocity_z": velocity_z,
                "speed_mps": speed_mps,
                "heading_deg": safe_float(row.get("heading")),
                "horizontal_accuracy": horizontal_accuracy,
                "vertical_accuracy": vertical_accuracy,
                "speed_accuracy": speed_accuracy,
                "flight_id": clean_text(row.get("flight_id")),
                "alt_source": alt_source,
                "split": paths.split,
                "dataset": paths.dataset,
                "scenario": paths.scenario,
                "iteration": paths.iteration,
                "split_group": paths.split_group,
                "source_file": path.name,
                "source_relpath": _relative(path, cfg),
                "raw_index": index,
            }
        )
    stat = {
        "split": paths.split,
        "dataset": paths.dataset,
        "scenario": paths.scenario,
        "iteration": paths.iteration,
        "split_group": paths.split_group,
        "source_kind": "ground_truth",
        "source_file": path.name,
        "source_relpath": _relative(path, cfg),
        "sensor_type_raw": "GROUND-TRUTH",
        "sensor_type": "ground_truth",
        "sensor_code": None,
        "platform": None,
        "node_ids": None,
        "n_rows_raw": len(rows) + malformed,
        "n_rows_parsed": len(records),
        "n_rows_malformed": malformed,
        "n_rows_missing_ts": missing_ts,
        "n_rows_missing_sensor_loc": 0,
        "n_rows_with_location": len(records),
        "n_rows_with_range_bearing": 0,
        "n_rows_out": len(records),
        "t_min_iso": ms_to_iso(t_min) if t_min is not None else ms_to_iso(file_meta["file_start_ms"]),
        "t_max_iso": ms_to_iso(t_max) if t_max is not None else ms_to_iso(file_meta["file_end_ms"]),
        "t_min_ms": t_min if t_min is not None else file_meta["file_start_ms"],
        "t_max_ms": t_max if t_max is not None else file_meta["file_end_ms"],
        "file_start_iso": ms_to_iso(file_meta["file_start_ms"]),
        "file_end_iso": ms_to_iso(file_meta["file_end_ms"]),
        "file_start_ms": file_meta["file_start_ms"],
        "file_end_ms": file_meta["file_end_ms"],
        "coord_systems": None,
        "datums": None,
        "columns": "|".join(header),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "model_raw": model_raw,
        "color": file_meta["color"],
        "n_serials": 0,
    }
    if malformed:
        issues.append(
            {
                "split_group": paths.split_group,
                "source_file": path.name,
                "source_relpath": _relative(path, cfg),
                "raw_index": None,
                "issue": "malformed_gt_csv_row",
                "detail": f"{malformed} rows with field count mismatch",
            }
        )
    return records, stat, issues


def parse_friendly_file(
    path: Path,
    paths: IterationPaths,
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parsea la lista Excel de drones amigos."""
    serials, records = load_friendly_serials(path)
    output: list[dict[str, Any]] = []
    for record in records:
        output.append(
            {
                "split": paths.split,
                "dataset": paths.dataset,
                "scenario": paths.scenario,
                "iteration": paths.iteration,
                "split_group": paths.split_group,
                "manufacturer": record["manufacturer"],
                "serial_raw": record["serial_raw"],
                "serial_norm": record["serial_norm"],
                "source_file": path.name,
                "source_relpath": _relative(path, cfg),
                "raw_index": record["raw_index"],
            }
        )
    stat = {
        "split": paths.split,
        "dataset": paths.dataset,
        "scenario": paths.scenario,
        "iteration": paths.iteration,
        "split_group": paths.split_group,
        "source_kind": "friendly_list",
        "source_file": path.name,
        "source_relpath": _relative(path, cfg),
        "sensor_type_raw": "FRIENDLY-LIST",
        "sensor_type": "friendly_list",
        "sensor_code": None,
        "platform": None,
        "node_ids": None,
        "n_rows_raw": len(records),
        "n_rows_parsed": len(records),
        "n_rows_malformed": 0,
        "n_rows_missing_ts": 0,
        "n_rows_missing_sensor_loc": 0,
        "n_rows_with_location": 0,
        "n_rows_with_range_bearing": 0,
        "n_rows_out": len(records),
        "t_min_iso": None,
        "t_max_iso": None,
        "t_min_ms": None,
        "t_max_ms": None,
        "file_start_iso": None,
        "file_end_iso": None,
        "file_start_ms": None,
        "file_end_ms": None,
        "coord_systems": None,
        "datums": None,
        "columns": "Manufacturer|Serial Number",
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "model_raw": None,
        "color": None,
        "n_serials": len(serials),
    }
    return output, stat


def parse_template_inventory(
    path: Path,
    paths: IterationPaths,
    cfg: Config,
) -> dict[str, Any]:
    """Resume la plantilla Excel para el inventario."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook.active
    rows = worksheet.iter_rows(values_only=True)
    try:
        header = next(rows)
    except StopIteration:
        header = ()
    n_rows = 0
    t_min: int | None = None
    t_max: int | None = None
    for row in rows:
        n_rows += 1
        ts_ms = parse_iso_to_ms(row[0] if len(row) > 0 else None)
        if ts_ms is not None:
            if t_min is None or ts_ms < t_min:
                t_min = ts_ms
            if t_max is None or ts_ms > t_max:
                t_max = ts_ms
    workbook.close()
    return {
        "split": paths.split,
        "dataset": paths.dataset,
        "scenario": paths.scenario,
        "iteration": paths.iteration,
        "split_group": paths.split_group,
        "source_kind": "template",
        "source_file": path.name,
        "source_relpath": _relative(path, cfg),
        "sensor_type_raw": "TEMPLATE",
        "sensor_type": "template",
        "sensor_code": None,
        "platform": None,
        "node_ids": None,
        "n_rows_raw": n_rows,
        "n_rows_parsed": n_rows,
        "n_rows_malformed": 0,
        "n_rows_missing_ts": 0,
        "n_rows_missing_sensor_loc": 0,
        "n_rows_with_location": 0,
        "n_rows_with_range_bearing": 0,
        "n_rows_out": n_rows,
        "t_min_iso": ms_to_iso(t_min),
        "t_max_iso": ms_to_iso(t_max),
        "t_min_ms": t_min,
        "t_max_ms": t_max,
        "file_start_iso": None,
        "file_end_iso": None,
        "file_start_ms": None,
        "file_end_ms": None,
        "coord_systems": None,
        "datums": None,
        "columns": "|".join(str(value) for value in header),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "model_raw": None,
        "color": None,
        "n_serials": 0,
    }


def build_origins(detections: pd.DataFrame, cfg: Config) -> dict[str, dict[str, Any]]:
    """Calcula el origen ENU local de cada `split_group`."""
    enu_cfg = cfg.etl.get("enu", {})
    fallback = enu_cfg.get("fallback_origin", {})
    fallback_tuple = (
        float(fallback.get("lat", 51.52)),
        float(fallback.get("lon", 5.86)),
        float(fallback.get("alt_m", 0.0)),
    )
    mode = str(enu_cfg.get("origin_mode", "median_sensor_location"))
    origins: dict[str, dict[str, Any]] = {}
    if detections.empty:
        return origins
    for group, part in detections.groupby("split_group", sort=True):
        origin = None
        source = None
        count = 0
        if mode == "median_sensor_location":
            sensors = part.loc[
                part["has_sensor_location"],
                ["sensor_id", "sensor_lat", "sensor_lon", "sensor_alt_m"],
            ].drop_duplicates()
            lat, lon, alt, count = median_origin(
                sensors["sensor_lat"],
                sensors["sensor_lon"],
                sensors["sensor_alt_m"],
                fallback_tuple,
            )
            if count > 0:
                origin = (lat, lon, alt)
                source = "sensor_median"
        if origin is None:
            lat, lon, alt, count = median_origin(
                part["raw_lat"],
                part["raw_lon"],
                part["raw_alt_m"],
                fallback_tuple,
            )
            if count > 0:
                origin = (lat, lon, alt)
                source = "object_median"
        if origin is None:
            origin = fallback_tuple
            source = "fallback"
            count = 0
        origins[str(group)] = {
            "lat": origin[0],
            "lon": origin[1],
            "alt_m": origin[2],
            "source": source,
            "n_points": count,
        }
    return origins


def add_enu(
    detections: pd.DataFrame,
    ground_truth: pd.DataFrame,
    cfg: Config,
    origins: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Añade coordenadas ENU a detecciones y ground truth."""
    fallback = cfg.etl.get("enu", {}).get("fallback_origin", {})
    fallback_origin = (
        float(fallback.get("lat", 51.52)),
        float(fallback.get("lon", 5.86)),
        float(fallback.get("alt_m", 0.0)),
    )
    if not detections.empty:
        for group, indexes in detections.groupby("split_group").groups.items():
            origin_data = origins.get(str(group), {})
            origin = (
                float(origin_data.get("lat", fallback_origin[0])),
                float(origin_data.get("lon", fallback_origin[1])),
                float(origin_data.get("alt_m", fallback_origin[2])),
            )
            part = detections.loc[indexes]
            alt_for_enu = part["alt_m"].fillna(origin[2])
            east, north, up = geodetic_to_enu(
                part["lat"], part["lon"], alt_for_enu, origin[0], origin[1], origin[2]
            )
            detections.loc[indexes, "n"] = north
            detections.loc[indexes, "e"] = east
            detections.loc[indexes, "u"] = up
            detections.loc[part.index[~part["has_alt"].to_numpy()], "u"] = np.nan
            sensor_alt_for_enu = part["sensor_alt_m"].fillna(origin[2])
            east, north, up = geodetic_to_enu(
                part["sensor_lat"],
                part["sensor_lon"],
                sensor_alt_for_enu,
                origin[0],
                origin[1],
                origin[2],
            )
            detections.loc[indexes, "sensor_n"] = north
            detections.loc[indexes, "sensor_e"] = east
            detections.loc[indexes, "sensor_u"] = up
            detections.loc[part.index[part["sensor_alt_m"].isna().to_numpy()], "sensor_u"] = np.nan
            raw_alt_for_enu = part["raw_alt_m"].fillna(origin[2])
            east, north, up = geodetic_to_enu(
                part["raw_lat"],
                part["raw_lon"],
                raw_alt_for_enu,
                origin[0],
                origin[1],
                origin[2],
            )
            detections.loc[indexes, "raw_n"] = north
            detections.loc[indexes, "raw_e"] = east
            detections.loc[indexes, "raw_u"] = up
            detections.loc[part.index[part["raw_alt_m"].isna().to_numpy()], "raw_u"] = np.nan
    if not ground_truth.empty:
        for group, indexes in ground_truth.groupby("split_group").groups.items():
            origin_data = origins.get(str(group), {})
            origin = (
                float(origin_data.get("lat", fallback_origin[0])),
                float(origin_data.get("lon", fallback_origin[1])),
                float(origin_data.get("alt_m", fallback_origin[2])),
            )
            part = ground_truth.loc[indexes]
            east, north, up = geodetic_to_enu(
                part["lat"], part["lon"], part["alt_m"], origin[0], origin[1], origin[2]
            )
            ground_truth.loc[indexes, "n"] = north
            ground_truth.loc[indexes, "e"] = east
            ground_truth.loc[indexes, "u"] = up
    return detections, ground_truth


def deduplicate(detections: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, int]:
    """Elimina duplicados por `row_uid` y devuelve cuántos quitó."""
    if detections.empty or not bool(cfg.etl.get("deduplicate", True)):
        return detections, 0
    before = len(detections)
    deduplicated = detections.drop_duplicates(subset=["row_uid"], keep="first").copy()
    return deduplicated, before - len(deduplicated)


def _write_parquet(frame: pd.DataFrame, path: Path, cfg: Config) -> None:
    compression = str(cfg.etl.get("parquet_compression", "zstd"))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression=compression)
    if bool(cfg.etl.get("write_csv_copy", False)):
        frame.to_csv(path.with_suffix(".csv"), index=False)


def _table_quality(
    table: str,
    frame: pd.DataFrame,
    key_columns: list[str],
) -> dict[str, Any]:
    rows = int(len(frame))
    columns = int(frame.shape[1])
    null_cells = int(frame.isna().sum().sum()) if rows else 0
    return {
        "rows": rows,
        "columns": columns,
        "null_cells": null_cells,
        "null_cells_pct": round(100.0 * null_cells / (rows * columns), 4) if rows and columns else 0.0,
        "rows_with_any_null": int(frame.isna().any(axis=1).sum()) if rows else 0,
        "key_null_counts": {
            column: int(frame[column].isna().sum())
            for column in key_columns
            if column in frame.columns
        },
    }


def _null_report(table: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_rows = int(len(frame))
    for column in frame.columns:
        n_null = int(frame[column].isna().sum())
        rows.append(
            {
                "table": table,
                "column": column,
                "dtype": str(frame[column].dtype),
                "n_rows": n_rows,
                "n_null": n_null,
                "pct_null": round(100.0 * n_null / n_rows, 4) if n_rows else 0.0,
                "n_non_null": n_rows - n_null,
            }
        )
    return rows


def _quality(result: EtlResult) -> dict[str, Any]:
    detections = result.detections
    gt = result.ground_truth
    friendly = result.friendly
    inventory = result.inventory
    issues = result.issues
    detection_keys = [
        "lat",
        "lon",
        "alt_m",
        "n",
        "e",
        "u",
        "sigma_n",
        "sigma_e",
        "sigma_u",
        "az_deg",
        "el_deg",
        "class_raw",
        "class_conf",
        "serial",
        "model",
        "protocol",
        "freq_mhz",
    ]
    gt_keys = [
        "lat",
        "lon",
        "alt_m",
        "n",
        "e",
        "u",
        "sigma_n",
        "sigma_e",
        "sigma_u",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "heading_deg",
        "flight_id",
    ]
    detection_quality = _table_quality("detections", detections, detection_keys)
    detection_quality.update(
        {
            "rows_without_position": int((~detections["has_position"]).sum()) if not detections.empty else 0,
            "rows_without_alt": int((~detections["has_alt"]).sum()) if not detections.empty else 0,
            "rows_without_bearing": int((~detections["has_bearing"]).sum()) if not detections.empty else 0,
            "rows_without_class": int((~detections["has_class"]).sum()) if not detections.empty else 0,
            "rows_without_serial": int((~detections["has_serial"]).sum()) if not detections.empty else 0,
            "rows_without_metadata": int((~detections["has_metadata"]).sum()) if not detections.empty else 0,
            "rows_without_velocity": int((~detections["has_velocity"]).sum()) if not detections.empty else 0,
            "rows_without_sensor_location": int(detections["sensor_lat"].isna().sum()) if not detections.empty else 0,
            "rows_without_timestamp": int(detections["ts_ms"].isna().sum()) if not detections.empty else 0,
        }
    )
    gt_quality = _table_quality("ground_truth", gt, gt_keys)
    gt_quality.update(
        {
            "rows_geo_altitude": int((gt["alt_source"] == "geo_altitude").sum()) if not gt.empty else 0,
            "rows_altitude_fallback": int((gt["alt_source"] == "altitude_fallback").sum()) if not gt.empty else 0,
            "rows_missing_alt": int((gt["alt_source"] == "missing").sum()) if not gt.empty else 0,
        }
    )
    friendly_quality = _table_quality(
        "friendly_uas", friendly, ["manufacturer", "serial_raw", "serial_norm"]
    )
    sensor_rows = inventory[inventory["source_kind"] == "sensor"] if not inventory.empty else inventory
    gt_rows = inventory[inventory["source_kind"] == "ground_truth"] if not inventory.empty else inventory
    friendly_rows = inventory[inventory["source_kind"] == "friendly_list"] if not inventory.empty else inventory
    issue_counts = (
        {str(key): int(value) for key, value in issues.groupby("issue").size().to_dict().items()}
        if not issues.empty
        else {}
    )
    return {
        "detections": detection_quality,
        "ground_truth": gt_quality,
        "friendly_uas": friendly_quality,
        "duplicates": {
            "removed": int(result.summary.get("n_duplicates_removed", 0)),
            "remaining": int(detections.duplicated(subset=["row_uid"]).sum()) if not detections.empty else 0,
        },
        "rows_in": {
            "detections": int(sensor_rows["n_rows_raw"].sum()) if not sensor_rows.empty else 0,
            "ground_truth": int(gt_rows["n_rows_raw"].sum()) if not gt_rows.empty else 0,
            "friendly_uas": int(friendly_rows["n_rows_raw"].sum()) if not friendly_rows.empty else 0,
        },
        "rows_dropped": {
            "malformed_csv": int(sensor_rows["n_rows_malformed"].sum()) if not sensor_rows.empty else 0,
            "missing_timestamp": int((issues["issue"] == "missing_or_invalid_timestamp").sum()) if not issues.empty else 0,
            "missing_gt_timestamp": int((issues["issue"] == "missing_or_invalid_gt_timestamp").sum()) if not issues.empty else 0,
            "missing_gt_location": int((issues["issue"] == "missing_or_invalid_gt_location").sum()) if not issues.empty else 0,
        },
        "issues_by_type": issue_counts,
    }


def _summary(result: EtlResult) -> dict[str, Any]:
    detections = result.detections
    gt = result.ground_truth
    friendly = result.friendly
    by_split = detections.groupby("split").size().to_dict() if not detections.empty else {}
    by_sensor = detections.groupby("sensor_type").size().to_dict() if not detections.empty else {}
    by_class = detections.groupby("class_norm").size().to_dict() if not detections.empty else {}
    by_top = detections.groupby("class_top").size().to_dict() if not detections.empty else {}
    by_pos = detections.groupby("pos_source").size().to_dict() if not detections.empty else {}
    by_sigma = detections.groupby("sigma_source").size().to_dict() if not detections.empty else {}
    by_group = detections.groupby("split_group").size().to_dict() if not detections.empty else {}
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_roots": {
            "test": str(result.summary.get("raw_test", "")),
            "eval": str(result.summary.get("raw_eval", "")),
        },
        "counts": {
            "iterations": int(result.summary.get("n_iterations", 0)),
            "sensor_files": int(result.summary.get("n_sensor_files", 0)),
            "gt_files": int(result.summary.get("n_gt_files", 0)),
            "friendly_files": int(result.summary.get("n_friendly_files", 0)),
            "template_files": int(result.summary.get("n_template_files", 0)),
            "detections_out": int(len(detections)),
            "detections_duplicates_removed": int(result.summary.get("n_duplicates_removed", 0)),
            "ground_truth_out": int(len(gt)),
            "friendly_out": int(len(friendly)),
            "issues": int(len(result.issues)),
        },
        "by_split": {str(key): int(value) for key, value in by_split.items()},
        "by_split_group": {str(key): int(value) for key, value in by_group.items()},
        "by_sensor_type": {str(key): int(value) for key, value in by_sensor.items()},
        "by_class_norm": {str(key): int(value) for key, value in by_class.items()},
        "by_class_top": {str(key): int(value) for key, value in by_top.items()},
        "by_pos_source": {str(key): int(value) for key, value in by_pos.items()},
        "by_sigma_source": {str(key): int(value) for key, value in by_sigma.items()},
        "data_quality": _quality(result),
        "origins": result.origins,
        "tables": {
            "detections_rows": int(len(detections)),
            "ground_truth_rows": int(len(gt)),
            "friendly_rows": int(len(friendly)),
        },
    }


def run_etl(cfg: Config, write_interim: bool = True) -> EtlResult:
    """Ejecuta el ETL completo y opcionalmente escribe `data/interim/`."""
    cfg.ensure_dirs()
    iterations = discover_iterations(cfg)
    detection_records: list[dict[str, Any]] = []
    gt_records: list[dict[str, Any]] = []
    friendly_records: list[dict[str, Any]] = []
    inventory_records: list[dict[str, Any]] = []
    issue_records: list[dict[str, Any]] = []
    n_sensor_files = 0
    n_gt_files = 0
    n_friendly_files = 0
    n_template_files = 0

    for paths in iterations:
        friendly_serials: set[str] | None = None
        if paths.friendly_files:
            friendly_serials = set()
            for friendly_file in paths.friendly_files:
                records, stat = parse_friendly_file(friendly_file, paths, cfg)
                serials = {record["serial_norm"] for record in records}
                friendly_serials |= serials
                friendly_records.extend(records)
                inventory_records.append(stat)
                n_friendly_files += 1
        for sensor_file in paths.sensor_files:
            records, stat, issues = parse_sensor_file(
                sensor_file, paths, cfg, friendly_serials
            )
            detection_records.extend(records)
            inventory_records.append(stat)
            issue_records.extend(issues)
            n_sensor_files += 1
        for gt_file in paths.gt_files:
            records, stat, issues = parse_gt_file(gt_file, paths, cfg)
            gt_records.extend(records)
            inventory_records.append(stat)
            issue_records.extend(issues)
            n_gt_files += 1
        if paths.template_file is not None:
            inventory_records.append(
                parse_template_inventory(paths.template_file, paths, cfg)
            )
            n_template_files += 1

    detections = pd.DataFrame(detection_records).reindex(columns=DETECTION_COLUMNS)
    ground_truth = pd.DataFrame(gt_records).reindex(columns=GT_COLUMNS)
    friendly = pd.DataFrame(friendly_records).reindex(columns=FRIENDLY_COLUMNS)
    inventory = pd.DataFrame(inventory_records).reindex(columns=INVENTORY_COLUMNS)
    issues = pd.DataFrame(issue_records).reindex(
        columns=["split_group", "source_file", "source_relpath", "raw_index", "issue", "detail"]
    )

    if not bool(cfg.etl.get("keep_non_drone", True)) and not detections.empty:
        detections = detections[~detections["class_is_non_drone"]].copy()

    if not detections.empty:
        detections["ts_ms"] = detections["ts_ms"].astype("int64")
        detections["scenario"] = detections["scenario"].astype("int16")
        detections["iteration"] = detections["iteration"].astype("int16")
        detections = detections.sort_values(
            ["split", "split_group", "ts_ms", "sensor_id", "raw_index"]
        ).reset_index(drop=True)
    if not ground_truth.empty:
        ground_truth["ts_ms"] = ground_truth["ts_ms"].astype("int64")
        ground_truth["scenario"] = ground_truth["scenario"].astype("int16")
        ground_truth["iteration"] = ground_truth["iteration"].astype("int16")
        ground_truth = ground_truth.sort_values(
            ["split", "split_group", "ts_ms", "drone_key", "raw_index"]
        ).reset_index(drop=True)
    if not friendly.empty:
        friendly["scenario"] = friendly["scenario"].astype("int16")
        friendly["iteration"] = friendly["iteration"].astype("int16")
        friendly = friendly.sort_values(
            ["split", "split_group", "manufacturer", "serial_norm"]
        ).reset_index(drop=True)

    origins = build_origins(detections, cfg)
    detections, ground_truth = add_enu(detections, ground_truth, cfg, origins)
    detections, duplicates_removed = deduplicate(detections, cfg)
    detections = detections.sort_values(
        ["split", "split_group", "ts_ms", "sensor_id", "raw_index"]
    ).reset_index(drop=True)

    result = EtlResult(
        detections=detections,
        ground_truth=ground_truth,
        friendly=friendly,
        inventory=inventory,
        issues=issues,
        origins=origins,
        summary={
            "raw_test": str(cfg.raw_test),
            "raw_eval": str(cfg.raw_eval),
            "n_iterations": len(iterations),
            "n_sensor_files": n_sensor_files,
            "n_gt_files": n_gt_files,
            "n_friendly_files": n_friendly_files,
            "n_template_files": n_template_files,
            "n_duplicates_removed": duplicates_removed,
        },
    )
    summary = _summary(result)
    result.summary = summary

    if write_interim:
        _write_parquet(detections, cfg.interim / "detections.parquet", cfg)
        _write_parquet(ground_truth, cfg.interim / "ground_truth.parquet", cfg)
        _write_parquet(friendly, cfg.interim / "friendly_uas.parquet", cfg)
        for group in sorted(detections["split_group"].unique()) if not detections.empty else []:
            subset = detections[detections["split_group"] == group]
            split = str(subset["split"].iloc[0])
            name = _sanitize(group)
            _write_parquet(subset, cfg.interim / split / f"{name}_detections.parquet", cfg)
        for group in sorted(ground_truth["split_group"].unique()) if not ground_truth.empty else []:
            subset = ground_truth[ground_truth["split_group"] == group]
            split = str(subset["split"].iloc[0])
            name = _sanitize(group)
            _write_parquet(subset, cfg.interim / split / f"{name}_ground_truth.parquet", cfg)
        with open(cfg.interim / "origins.json", "w", encoding="utf-8") as handle:
            json.dump(json_safe(origins), handle, indent=2, ensure_ascii=False)
        with open(cfg.interim / "schema.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "detections": {column: str(dtype) for column, dtype in detections.dtypes.items()},
                    "ground_truth": {column: str(dtype) for column, dtype in ground_truth.dtypes.items()},
                    "friendly_uas": {column: str(dtype) for column, dtype in friendly.dtypes.items()},
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    inventory.to_csv(cfg.reports / "inventory.csv", index=False)
    issues.to_csv(cfg.reports / "etl_issues.csv", index=False)
    null_rows = (
        _null_report("detections", detections)
        + _null_report("ground_truth", ground_truth)
        + _null_report("friendly_uas", friendly)
    )
    pd.DataFrame(null_rows).to_csv(cfg.reports / "null_counts.csv", index=False)
    sensor_files = inventory[inventory["source_kind"] == "sensor"]
    sensor_files.to_csv(cfg.reports / "sensor_file_summary.csv", index=False)
    with open(cfg.reports / "etl_summary.json", "w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2, ensure_ascii=False)
    return result
