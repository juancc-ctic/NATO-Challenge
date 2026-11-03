"""Validación de las plantillas Excel de entrega.

Existe porque el evaluador espera 19 columnas y los timestamps no se deben
tocar; por eso valida la cabecera y copia `timestamp`/`unix_timestamp`.
"""
from __future__ import annotations

import json
import re
from typing import Any

from openpyxl import Workbook, load_workbook

from src.config import Config
from src.etl import discover_iterations
from src.parsing import json_safe
from src.schema import TEMPLATE_COLUMNS


def _name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def run_check(cfg: Config) -> dict[str, Any]:
    """Valida todas las plantillas y escribe `reports/check.json` y los Excel de comprobación."""
    cfg.ensure_dirs()
    iterations = discover_iterations(cfg)
    results: list[dict[str, Any]] = []
    for paths in iterations:
        if paths.template_file is None:
            results.append(
                {
                    "split_group": paths.split_group,
                    "template": None,
                    "output": None,
                    "status": "missing_template",
                }
            )
            continue
        source = load_workbook(paths.template_file, read_only=True, data_only=True)
        worksheet = source.active
        rows = worksheet.iter_rows(values_only=True)
        try:
            header = next(rows)
        except StopIteration:
            header = ()
        header_ok = tuple(header) == tuple(TEMPLATE_COLUMNS)
        n_rows = 0
        missing_timestamp = 0
        missing_unix_timestamp = 0
        output_workbook = Workbook(write_only=True)
        output_sheet = output_workbook.create_sheet()
        output_sheet.append(list(TEMPLATE_COLUMNS))
        for row in rows:
            n_rows += 1
            timestamp = row[0] if len(row) > 0 else None
            unix_timestamp = row[1] if len(row) > 1 else None
            if timestamp in (None, ""):
                missing_timestamp += 1
            if unix_timestamp in (None, ""):
                missing_unix_timestamp += 1
            output_sheet.append(
                [timestamp, unix_timestamp] + [None] * (len(TEMPLATE_COLUMNS) - 2)
            )
        output_dir = cfg.reports / "check"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"_check_{_name(paths.split_group)}.xlsx"
        output_workbook.save(output)
        source.close()
        results.append(
            {
                "split_group": paths.split_group,
                "template": str(paths.template_file),
                "output": str(output),
                "status": "ok" if header_ok else "header_mismatch",
                "n_rows": n_rows,
                "n_columns": len(header),
                "header_ok": bool(header_ok),
                "missing_timestamp": missing_timestamp,
                "missing_unix_timestamp": missing_unix_timestamp,
            }
        )
    document = {
        "n_iterations": len(iterations),
        "n_templates": sum(1 for item in results if item["template"] is not None),
        "n_ok": sum(1 for item in results if item["status"] == "ok"),
        "results": results,
    }
    with open(cfg.reports / "check.json", "w", encoding="utf-8") as handle:
        json.dump(json_safe(document), handle, indent=2, ensure_ascii=False)
    return document
