"""Normalización de clasificaciones de sensores.

Existe porque cada sensor reporta clases anidadas y con distinta confianza;
por eso aplana `classification` y la mapea al vocabulario cerrado del Excel.
"""
from __future__ import annotations

import re
from typing import Any

from src.metadata import meta_items
from src.parsing import clean_text

NON_DRONE_TOKENS = (
    "bird",
    "animal",
    "land vehicle",
    "sea vessel",
    "fixed object",
    "home",
    "person",
    "human",
    "car",
)

FIXED_WING_TOKENS = ("fixed wing", "fixedwing", "fixed_wing")
ROTARY_WING_TOKENS = ("rotary wing", "rotarywing", "rotary_wing", "multirotor", "multi rotor")
TRANSMITTER_TOKENS = ("transmitter", "em transmitter")
AIR_VEHICLE_TOKENS = ("air vehicle", "airvehicle", "aircraft", "air craft", "uav")


def _key(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def flatten_classification(raw: Any) -> tuple[str, float, list[str], str, float]:
    """Aplana `classification` y devuelve hoja, confianza, ruta y tipo superior."""
    items = meta_items(raw)
    nodes: list[tuple[int, str, float, list[str]]] = []

    def walk(entries: list[Any], depth: int, path: list[str]) -> None:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = clean_text(entry.get("type"))
            if not name:
                continue
            confidence = entry.get("confidence")
            try:
                confidence_value = float(confidence) if confidence is not None else 1.0
            except (TypeError, ValueError):
                confidence_value = 1.0
            nodes.append((depth, name, confidence_value, path + [name]))
            sub = entry.get("subClass")
            if isinstance(sub, list):
                walk(sub, depth + 1, path + [name])

    walk(items, 0, [])
    if not nodes:
        text = clean_text(raw)
        if text and not text.startswith(("[", "{", "(")):
            return text, 0.0, [text], text, 0.0
        return "Unknown", 0.0, [], "Unknown", 0.0
    max_depth = max(node[0] for node in nodes)
    leaves = [node for node in nodes if node[0] == max_depth]
    leaf = max(leaves, key=lambda node: node[2])
    top = nodes[0]
    return leaf[1], leaf[2], leaf[3], top[1], top[2]


def map_class(class_raw: str, class_path: list[str], class_top: str) -> tuple[str, bool, bool, bool]:
    """Mapea la clase a las etiquetas del Excel y flags dron/no-dron/transmisor."""
    leaf = _key(class_raw)
    top = _key(class_top)
    path = " | ".join(_key(item) for item in class_path)
    combined = " ".join([leaf, top, path])

    if any(token in combined for token in NON_DRONE_TOKENS):
        return "Non-drone", False, True, False
    if any(token in leaf for token in FIXED_WING_TOKENS) or any(token in path for token in FIXED_WING_TOKENS):
        return "Fixed Wing UAS", True, False, False
    if any(token in leaf for token in ROTARY_WING_TOKENS) or any(token in path for token in ROTARY_WING_TOKENS):
        return "Rotary Wing UAS", True, False, False
    if any(token in leaf for token in TRANSMITTER_TOKENS) or any(token in path for token in TRANSMITTER_TOKENS):
        return "UAS Transmitter", False, False, True
    if any(token in top for token in AIR_VEHICLE_TOKENS):
        return "Unknown", True, False, False
    return "Unknown", False, False, False
