"""Utilidades de parseo: CSV tolerante, timestamps ISO, números, JSON y hashes.

Existe porque los CSV traen JSON con comillas simples y filas mal formadas;
por eso lee de forma tolerante y convierte a tipos seguros.
"""
from __future__ import annotations

import csv
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a", "-"}:
        return math.nan
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    if not math.isfinite(number):
        return None
    return int(number)


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_iso_to_ms(value: Any) -> int | None:
    """Convierte un timestamp ISO 8601 a Unix ms UTC."""
    text = clean_text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000.0)


def ms_to_iso(ts_ms: Any) -> str | None:
    try:
        number = float(ts_ms)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    parsed = datetime.fromtimestamp(number / 1000.0, tz=timezone.utc)
    return parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def split_tolerant(line: str) -> list[str]:
    text = line.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    fields: list[str] = []
    buffer: list[str] = []
    depth = 0
    in_quote = False
    escaped = False
    for char in text:
        if escaped:
            buffer.append(char)
            escaped = False
            continue
        if char == "\\":
            buffer.append(char)
            escaped = True
            continue
        if char == '"' and depth == 0:
            in_quote = not in_quote
            buffer.append(char)
            continue
        if char in "[{(":
            depth += 1
        elif char in "]})":
            depth = max(0, depth - 1)
        if char == "," and depth == 0 and not in_quote:
            fields.append("".join(buffer))
            buffer = []
        else:
            buffer.append(char)
    fields.append("".join(buffer))
    result: list[str] = []
    for field in fields:
        item = field.strip()
        if len(item) >= 2 and item[0] == '"' and item[-1] == '"':
            item = item[1:-1]
        result.append(item)
    return result


def read_csv_tolerant(path: Path) -> tuple[list[str], list[dict[str, str]], int]:
    """Lee un CSV tolerando comillas, anidamientos y filas malformadas."""
    header: list[str] = []
    rows: list[dict[str, str]] = []
    malformed = 0
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [str(name).strip() for name in next(reader)]
        except StopIteration:
            return header, rows, malformed
        for raw in reader:
            if not raw or all(not str(value).strip() for value in raw):
                continue
            if len(raw) == len(header):
                values = raw
            elif len(raw) == 1:
                values = split_tolerant(raw[0])
            else:
                values = split_tolerant(",".join(raw))
            if len(values) == len(header):
                rows.append(dict(zip(header, values)))
            else:
                malformed += 1
    return header, rows, malformed


def parse_sensor_filename(path: Path) -> dict[str, Any]:
    """Extrae plataforma, sensor, tipo y fechas del nombre de un CSV de sensor."""
    stem = path.stem
    parts = stem.split("_", 5)
    if len(parts) == 6:
        platform, sensor_code, sensor_type_raw, date, start, end = parts
    else:
        platform = parts[0] if parts else "Unknown"
        sensor_code = parts[1] if len(parts) > 1 else "Unknown"
        sensor_type_raw = parts[2] if len(parts) > 2 else "Unknown"
        date = parts[3] if len(parts) > 3 else ""
        start = parts[4] if len(parts) > 4 else ""
        end = parts[5] if len(parts) > 5 else ""
    start_ms = None
    end_ms = None
    if date and start:
        try:
            start_ms = int(datetime.strptime(date + start, "%y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000.0)
        except ValueError:
            start_ms = None
    if date and end:
        try:
            end_ms = int(datetime.strptime(date + end, "%y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000.0)
        except ValueError:
            end_ms = None
    return {
        "platform": platform,
        "sensor_code": sensor_code,
        "sensor_type_raw": sensor_type_raw,
        "date": date,
        "start": start,
        "end": end,
        "file_start_ms": start_ms,
        "file_end_ms": end_ms,
    }


def parse_gt_filename(path: Path) -> dict[str, Any]:
    """Extrae color, fechas y modelo del nombre de un CSV Dronetag."""
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 6 and parts[0].lower() == "dronetag":
        color = parts[1].upper()
        date = parts[2]
        start = parts[3]
        end = parts[4]
        model_raw = "_".join(parts[5:]).strip() or "Unknown"
    else:
        color = "Unknown"
        date = ""
        start = ""
        end = ""
        model_raw = stem
    start_ms = None
    end_ms = None
    if date and start:
        try:
            start_ms = int(datetime.strptime(date + start, "%y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000.0)
        except ValueError:
            start_ms = None
    if date and end:
        try:
            end_ms = int(datetime.strptime(date + end, "%y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000.0)
        except ValueError:
            end_ms = None
    return {
        "color": color,
        "date": date,
        "start": start,
        "end": end,
        "model_raw": model_raw,
        "file_start_ms": start_ms,
        "file_end_ms": end_ms,
    }


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value
