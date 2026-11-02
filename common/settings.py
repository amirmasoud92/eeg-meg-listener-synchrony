"""Analysis parameters, data locations and the random streams used throughout."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class AnalysisConfig:
    project: SimpleNamespace
    dataset: SimpleNamespace
    sections: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        return self.sections[name]


def load_analysis(path: Path | None = None) -> AnalysisConfig:
    raw = _read_yaml(path or CONFIG_DIR / "analysis_parameters.yaml")
    sections = {k: v for k, v in raw.items() if k not in ("project", "dataset")}
    return AnalysisConfig(SimpleNamespace(**raw["project"]), SimpleNamespace(**raw["dataset"]), sections)


def load_paths(path: Path | None = None) -> SimpleNamespace:
    """Data locations from config/paths.yaml; relative paths resolve against the repository root."""
    raw = _read_yaml(path or CONFIG_DIR / "paths.yaml")
    aalborg = (PROJECT_ROOT / raw["raw_root"]).resolve()
    return SimpleNamespace(preprocessed_dir=aalborg / raw["preprocessed_dir"], processed_dir=aalborg / raw["processed_dir"],
                           channel_list=aalborg / raw["channel_list"], processed_root=(PROJECT_ROOT / raw["processed_root"]).resolve(),
                           derived_root=(PROJECT_ROOT / raw["derived_root"]).resolve(),
                           results_root=(PROJECT_ROOT / raw["results_root"]).resolve(),
                           mne_data=(PROJECT_ROOT / raw["mne_data"]).resolve())


def dataset_root(name: str) -> Path:
    """Folder of a public release, from the `datasets` entry of config/paths.yaml."""
    return (PROJECT_ROOT / _read_yaml(CONFIG_DIR / "paths.yaml")["datasets"][name]).resolve()


def rng(seed: int, *keys: int | str) -> np.random.Generator:
    """Independent, reproducible stream derived from a base seed and a key path. The base seeds and key strings
    throughout the code are stream labels and carry no other meaning; changing one changes the draws."""
    def code(k: int | str) -> int:
        if isinstance(k, (int, np.integer)):
            return int(k)
        return int.from_bytes(str(k).encode(), "little") % (2**32)

    entropy = [seed] + [code(k) for k in keys]
    return np.random.default_rng(np.random.SeedSequence(entropy))
