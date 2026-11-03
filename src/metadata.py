"""Extracción de metadatos de sensores.

Existe porque modelo, protocolo, serial y frecuencia están repartidos entre
`objectInfo`, `trackInfo`, `additionalInformation`, `signal` e `id`, y las
unidades no son homogéneas; por eso los busca y normaliza.
"""
from __future__ import annotations

import ast
import json
import math
import re
from functools import lru_cache
from typing import Any

from src.identify import looks_like_serial, normalize_serial
from src.parsing import clean_text, safe_float


def _norm_key(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


@lru_cache(maxsize=16384)
def parse_literal(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (list, dict)):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    for _ in range(3):
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            try:
                value = json.loads(text)
            except (ValueError, TypeError):
                return None
        if isinstance(value, str) and value != text and value.strip().startswith(("[", "{", "(")):
            text = value.strip()
            continue
        return value
    return None


def meta_items(raw: Any) -> list[dict[str, Any]]:
    value = parse_literal(raw)
    if isinstance(value, str):
        value = parse_literal(value)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def lookup_value(items: list[dict[str, Any]], *keys: str) -> Any:
    wanted = {_norm_key(key) for key in keys}
    for item in items:
        if _norm_key(item.get("type")) in wanted:
            return item.get("value")
    for item in items:
        for key, value in item.items():
            if _norm_key(key) in wanted:
                return value
    return None


_lookup = lookup_value


def parse_drone_event(items: list[dict[str, Any]]) -> dict[str, Any]:
    for item in items:
        if _norm_key(item.get("type")) != "drone event info":
            continue
        raw_value = item.get("value")
        if not isinstance(raw_value, str):
            continue
        event: dict[str, Any] = {}
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, dict):
                event = parsed
        except (ValueError, TypeError):
            event = {}
        if not event:
            for key in ("id", "serial_number", "factory_name", "can_takeover", "speed_ms"):
                match = re.search(rf"{key}\s*:\s*([^,}}]+)", raw_value)
                if match:
                    event[key] = match.group(1).strip().strip('"').strip("'")
        return event
    return {}


def infer_serial(value: Any) -> str | None:
    if looks_like_serial(value):
        return normalize_serial(value)
    return None


def looks_like_protocol(value: Any) -> bool:
    text = clean_text(value)
    if not text:
        return False
    upper = text.upper()
    if len(upper) > 40:
        return False
    return any(
        token in upper
        for token in ("OCUSYNC", "LIGHTBRIDGE", "WIFIBROADCAST", "WIFI", "ANALOG", "ELRS", "CROSSFIRE")
    )


def infer_manufacturer(model: Any) -> str | None:
    text = clean_text(model)
    if not text:
        return None
    lower = text.lower()
    if any(token in lower for token in ("dji", "mavic", "matrice", "phantom", "inspire", "mini", "air 2", "air2")):
        return "DJI"
    if any(token in lower for token in ("autel", "evo")):
        return "Autel"
    if any(token in lower for token in ("parrot", "anafi")):
        return "Parrot"
    if "skydio" in lower:
        return "Skydio"
    if any(token in lower for token in ("quantum", "vector")):
        return "Quantum Vector"
    if "heidrun" in lower:
        return "Heidrun"
    if "skyranger" in lower:
        return "Skyranger"
    return None


MODEL_ALIASES = {
    "mavic 3": "DJI Mavic 3",
    "dji mavic 3": "DJI Mavic 3",
    "matrice 300": "DJI Matrice 300",
    "dji matrice 300": "DJI Matrice 300",
    "autel evo2": "Autel Evo 2",
    "autel evo 2": "Autel Evo 2",
    "anafi": "Parrot Anafi",
    "parrot anafi": "Parrot Anafi",
    "vector": "Quantum Vector",
    "quantum vector": "Quantum Vector",
    "fpv": "FPV",
    "rq": "RQ",
    "skydio x10d": "Skydio X10D",
    "heidrun": "Heidrun",
    "skyranger": "Skyranger",
}


def normalize_model(model: Any) -> str | None:
    text = clean_text(model)
    if not text:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    key = text.lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    return text


def freq_to_mhz(value: Any) -> float:
    number = safe_float(value)
    if not math.isfinite(number) or number <= 0:
        return math.nan
    if number >= 1e8:
        return number / 1e6
    if number >= 1e5:
        return number / 1e3
    return number


def parse_freq(raw: Any) -> tuple[float, float]:
    items = meta_items(raw)
    values: list[float] = []
    for item in items:
        for key in ("centreFrequency", "centerFrequency", "frequency", "freq"):
            if key in item:
                value = freq_to_mhz(item.get(key))
                if math.isfinite(value):
                    values.append(value)
    if not values:
        return math.nan, math.nan
    return min(values), max(values)


def freq_band(freq_mhz: float, bands: list[dict[str, Any]]) -> str | None:
    if not math.isfinite(freq_mhz):
        return None
    for band in bands:
        if float(band["min"]) <= freq_mhz <= float(band["max"]):
            return str(band["name"])
    return f"{freq_mhz:.0f} MHz"


def extract_metadata(row: dict[str, Any], bands: list[dict[str, Any]]) -> dict[str, Any]:
    """Extrae serial, modelo, fabricante, protocolo, frecuencia y metadatos de una fila."""
    object_items = meta_items(row.get("objectInfo"))
    track_items = meta_items(row.get("trackInfo"))
    additional_items = meta_items(row.get("additionalInformation"))
    event = parse_drone_event(object_items)
    serial_raw = _lookup(object_items, "serial number", "serial_number", "serial", "drone serial")
    if not serial_raw:
        serial_raw = event.get("serial_number") or event.get("serial")
    if not serial_raw:
        serial_raw = infer_serial(row.get("id"))
    serial = normalize_serial(serial_raw)
    model_raw = (
        _lookup(object_items, "model", "uav model", "drone model")
        or event.get("factory_name")
        or _lookup(track_items, "model")
    )
    manufacturer = (
        _lookup(object_items, "droneMaker", "drone maker", "manufacturer", "factory name")
        or event.get("factory_name")
        or infer_manufacturer(model_raw)
    )
    protocol = (
        _lookup(object_items, "protocol", "technology", "transmission technology")
        or _lookup(track_items, "protocol")
        or _lookup(additional_items, "protocol")
    )
    if not protocol and looks_like_protocol(row.get("id")):
        protocol = clean_text(row.get("id"))
    freq_min, freq_max = parse_freq(row.get("signal"))
    freq_mhz = freq_max if math.isfinite(freq_max) else freq_min
    return {
        "serial": serial,
        "serial_raw": clean_text(serial_raw),
        "model": clean_text(model_raw),
        "model_norm": normalize_model(model_raw),
        "manufacturer": clean_text(manufacturer),
        "protocol": clean_text(protocol),
        "freq_mhz": freq_mhz,
        "freq_mhz_min": freq_min,
        "freq_mhz_max": freq_max,
        "freq_band": freq_band(freq_mhz, bands),
        "drone_mac": clean_text(_lookup(object_items, "droneMac", "drone mac")),
        "ap_mac": clean_text(_lookup(object_items, "apMac", "ap mac")),
        "pilot_mac": clean_text(_lookup(object_items, "pilotMac", "pilot mac")),
        "remote_id": clean_text(_lookup(object_items, "Remote ID operator ID", "operator id")),
        "heading_deg": safe_float(_lookup(object_items, "heading")),
        "channel": clean_text(_lookup(object_items, "channel")),
        "event_id": clean_text(event.get("id")),
        "event_serial": normalize_serial(event.get("serial_number")),
        "event_factory": clean_text(event.get("factory_name")),
        "event_takeover": clean_text(event.get("can_takeover")),
        "track_id_raw": clean_text(_lookup(track_items, "trackId", "track id")),
    }
