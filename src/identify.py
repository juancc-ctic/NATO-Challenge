"""Normalización de seriales y lectura/match de la lista de drones amigos.

Existe porque los seriales llegan con mayúsculas/espacios distintos y la lista
de amigos es un Excel; por eso normaliza y compara sin distinguir mayúsculas.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import openpyxl


def normalize_serial(value: Any) -> str | None:
    """Normaliza un serial para compararlo sin mayúsculas ni espacios."""
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    if not text or text.lower() in {"nan", "none", "null", "unknown", "not applicable", "n/a"}:
        return None
    text = re.sub(r"\s+", "", text)
    return text.upper()


def looks_like_serial(value: Any) -> bool:
    serial = normalize_serial(value)
    if serial is None:
        return False
    if len(serial) < 8 or len(serial) > 40:
        return False
    if any(char in serial for char in ".:/=+-"):
        return False
    if not re.fullmatch(r"[A-Z0-9]+", serial):
        return False
    if serial.isdigit():
        return len(serial) >= 12
    if any(token in serial for token in ("OCUSYNC", "LIGHTBRIDGE", "WIFIBROADCAST", "ANALOG", "CROSSFIRE", "ELRS")):
        return False
    return True


def _normalize_header(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def load_friendly_serials(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    """Lee la lista Excel de amigos y devuelve seriales y registros."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    serials: set[str] = set()
    records: list[dict[str, Any]] = []
    for sheet in workbook.worksheets:
        rows = sheet.iter_rows(values_only=True)
        try:
            header_row = next(rows)
        except StopIteration:
            continue
        header = [_normalize_header(value) for value in header_row]
        for index, row in enumerate(rows):
            values = dict(zip(header, row))
            manufacturer = values.get("manufacturer")
            serial_raw = values.get("serial number") or values.get("serial") or values.get("serialnumber")
            serial_norm = normalize_serial(serial_raw)
            if serial_norm is None:
                continue
            serials.add(serial_norm)
            records.append(
                {
                    "manufacturer": str(manufacturer).strip() if manufacturer is not None else None,
                    "serial_raw": str(serial_raw).strip() if serial_raw is not None else None,
                    "serial_norm": serial_norm,
                    "raw_index": index,
                }
            )
    workbook.close()
    return serials, records


def match_friendly(serial: Any, friendly_serials: set[str]) -> bool:
    """Indica si el serial está en la lista de amigos."""
    normalized = normalize_serial(serial)
    return normalized is not None and normalized in friendly_serials
