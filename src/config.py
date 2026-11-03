"""Carga `config.yaml` y resuelve las rutas del proyecto.

Existe porque las rutas y parámetros cambian entre máquinas; por eso no se
hardcodean y se leen de `config.yaml`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    path: Path
    root: Path
    raw_test: Path
    raw_eval: Path
    interim: Path
    reports: Path
    out: Path
    data: dict[str, Any]

    @property
    def etl(self) -> dict[str, Any]:
        return self.data.get("etl", {})

    @property
    def splits(self) -> dict[str, Any]:
        return self.data.get("splits", {})

    def ensure_dirs(self) -> None:
        for path in (
            self.interim,
            self.interim / "test",
            self.interim / "eval",
            self.reports,
            self.out,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path = "config.yaml") -> Config:
    """Carga el YAML y devuelve un `Config` con rutas resueltas."""
    config_path = Path(path).resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    project = data.get("project", {})
    root_value = project.get("root")
    root = (config_path.parent / str(root_value)).resolve() if root_value else config_path.parent
    raw = project.get("raw", {})
    return Config(
        path=config_path,
        root=root,
        raw_test=_resolve(root, raw.get("test", "data/raw/test")),
        raw_eval=_resolve(root, raw.get("eval", "data/raw/eval")),
        interim=_resolve(root, project.get("interim", "data/interim")),
        reports=_resolve(root, project.get("reports", "reports")),
        out=_resolve(root, project.get("out", "out")),
        data=data,
    )
