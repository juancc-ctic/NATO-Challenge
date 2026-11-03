"""Verifica la conversión interna -> Excel de entrega con tracks mockeados.

Recorre todas las plantillas, mockea 5 tracks por iteración, escribe un Excel
real y comprueba cabecera, timestamps, covarianza y round-trip ENU -> WGS84.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.etl import discover_iterations
from src.geo import covariance_neu_to_neu, enu_to_geodetic, geodetic_to_enu
from src.parsing import parse_iso_to_ms
from src.schema import TEMPLATE_COLUMNS
from src.submit import tracks_to_submission, write_submission


def _read_rows(path: Path) -> list[tuple]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    rows = list(workbook.active.iter_rows(values_only=True))
    workbook.close()
    return rows


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _mock_tracks(split_group: str, timestamps: list[int]) -> pd.DataFrame:
    north = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    east = np.array([0.0, 50.0, 100.0, 150.0, 200.0])
    up = np.array([50.0, 55.0, 60.0, 65.0, 70.0])
    covariance = np.array(
        [
            [4.0, 0.5, 0.1],
            [0.5, 9.0, 0.2],
            [0.1, 0.2, 16.0],
        ],
        dtype=float,
    )
    return pd.DataFrame(
        {
            "split_group": [split_group] * 5,
            "ts_ms": timestamps,
            "n": north,
            "e": east,
            "u": up,
            "C_nn": [covariance[0, 0]] * 5,
            "C_ne": [covariance[0, 1]] * 5,
            "C_nu": [covariance[0, 2]] * 5,
            "C_ee": [covariance[1, 1]] * 5,
            "C_eu": [covariance[1, 2]] * 5,
            "C_uu": [covariance[2, 2]] * 5,
            "track_id": [f"mock-{split_group}"] * 5,
            "classification": ["Rotary Wing UAS"] * 5,
            "classification_confidence": [0.9] * 5,
            "identity": ["Unknown"] * 5,
            "uav_model/manufacturer": ["DJI Mavic 3"] * 5,
            "protocol": ["OcuSync"] * 5,
            "serial_number": ["MOCK12345678"] * 5,
            "signal_frequency_band": ["2.4 GHz"] * 5,
        }
    )


def _verify_iteration(
    split_group: str,
    template: Path,
    origins: dict[str, dict],
    output_dir: Path,
) -> dict[str, float]:
    origin = origins[split_group]
    lat0 = float(origin["lat"])
    lon0 = float(origin["lon"])
    alt0 = float(origin["alt_m"])

    template_rows = _read_rows(template)
    template_data = template_rows[1:]
    sample_rows = template_data[:5]
    timestamps = [parse_iso_to_ms(row[0]) for row in sample_rows]
    if any(value is None for value in timestamps):
        raise AssertionError(f"Timestamp inválido en plantilla {split_group}")

    tracks = _mock_tracks(split_group, timestamps)
    submission = tracks_to_submission(tracks, origins)
    output = output_dir / f"mock_{_safe_name(split_group)}.xlsx"
    write_submission(submission, template, output)

    output_rows = _read_rows(output)
    if tuple(output_rows[0]) != tuple(TEMPLATE_COLUMNS):
        raise AssertionError(f"Cabecera incorrecta: {split_group}")
    if len(output_rows) - 1 != len(template_data):
        raise AssertionError(f"Filas distintas al template: {split_group}")

    for expected, actual in zip(template_data, output_rows[1:]):
        if expected[0] != actual[0] or expected[1] != actual[1]:
            raise AssertionError(f"Timestamp modificado: {split_group}")

    for index in range(5):
        row = output_rows[1 + index]
        if any(row[column] is None for column in (2, 3, 4, 11)):
            raise AssertionError(f"Fila mock incompleta: {split_group} {index}")
        cov = np.array(
            [
                [row[5], row[6], row[7]],
                [row[6], row[8], row[9]],
                [row[7], row[9], row[10]],
            ],
            dtype=float,
        )
        if not np.allclose(cov, cov.T):
            raise AssertionError(f"Covarianza no simétrica: {split_group} {index}")
        if np.linalg.eigvalsh(cov).min() < -1e-9:
            raise AssertionError(f"Covarianza no PSD: {split_group} {index}")
        e, n, u = geodetic_to_enu(
            row[2], row[3], row[4], lat0, lon0, alt0
        )
        if not (
            abs(e - tracks.iloc[index]["e"]) < 1e-6
            and abs(n - tracks.iloc[index]["n"]) < 1e-6
            and abs(u - tracks.iloc[index]["u"]) < 1e-6
        ):
            raise AssertionError(f"Round-trip ENU falla: {split_group} {index}")

    for row in output_rows[6:]:
        if any(value is not None for value in row[2:]):
            raise AssertionError(f"Filas no vacías fuera del mock: {split_group}")

    covariance = np.array(
        [
            [4.0, 0.5, 0.1],
            [0.5, 9.0, 0.2],
            [0.1, 0.2, 16.0],
        ],
        dtype=float,
    )
    same_origin = covariance_neu_to_neu(covariance, lat0, lon0, lat0, lon0)
    if not np.allclose(same_origin, covariance, atol=1e-12):
        raise AssertionError(f"Covarianza no se conserva: {split_group}")

    first = submission.iloc[0]
    moved = covariance_neu_to_neu(
        covariance, lat0, lon0, float(first["latitude"]), float(first["longitude"])
    )
    if not np.allclose(np.trace(moved), np.trace(covariance)):
        raise AssertionError(f"Traza no conservada: {split_group}")
    if not np.allclose(np.linalg.eigvalsh(moved), np.linalg.eigvalsh(covariance)):
        raise AssertionError(f"Autovalores no conservados: {split_group}")

    return {
        "split_group": split_group,
        "template": str(template),
        "output": str(output),
        "rows": len(output_rows) - 1,
        "trace": float(np.trace(covariance)),
        "eigenvalues": np.linalg.eigvalsh(covariance).tolist(),
    }


def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    origins = json.loads((ROOT / "data/interim/origins.json").read_text())
    output_dir = Path("/tmp/opencode/mock_submissions")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for iteration in discover_iterations(cfg):
        if iteration.template_file is None:
            raise AssertionError(f"Sin plantilla: {iteration.split_group}")
        results.append(
            _verify_iteration(
                iteration.split_group,
                iteration.template_file,
                origins,
                output_dir,
            )
        )

    print("PASS")
    for result in results:
        print(
            f"{result['split_group']}: {result['rows']} filas, "
            f"traza={result['trace']:.3f}, "
            f"autoval={np.round(result['eigenvalues'], 4).tolist()}"
        )
    print(f"salida: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
