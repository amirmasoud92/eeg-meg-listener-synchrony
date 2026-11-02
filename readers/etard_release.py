"""Readers for the Etard & Reichenbach (2019) release: BrainVision recordings, markers and stimulus files."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import yaml

from common.settings import CONFIG_DIR, PROJECT_ROOT

@dataclass(frozen=True)
class EtardConfig:
    raw: dict[str, Any]

    @property
    def root(self) -> Path:
        return (PROJECT_ROOT / self.raw["dataset"]["root"]).resolve()

    @property
    def dataset(self) -> dict[str, Any]:
        return self.raw["dataset"]

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]


def load_config(path: Path | None = None) -> EtardConfig:
    with open(path or CONFIG_DIR / "etard_parameters.yaml", "r", encoding="utf-8") as fh:
        return EtardConfig(yaml.safe_load(fh))


@dataclass(frozen=True)
class Marker:
    kind: str
    description: str
    position: int
    extra: str = ""


@dataclass
class Header:
    path: Path
    data_file: str
    marker_file: str
    n_channels: int
    sampling_interval_us: float
    binary_format: str
    orientation: str
    channels: list[str]
    resolutions: list[float]
    reference_name: str | None
    has_coordinates: bool

    @property
    def sfreq(self) -> float:
        return 1e6 / self.sampling_interval_us


def _ini_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            current = s[1:-1]
            sections[current] = []
        elif current:
            sections[current].append(line)
    return sections


def _kv(lines: list[str]) -> dict[str, str]:
    out = {}
    for line in lines:
        if "=" in line and not line.lstrip().startswith(";"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read_header(path: Path) -> Header:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    sec = _ini_sections(text)
    common = _kv(sec.get("Common Infos", []))
    binary = _kv(sec.get("Binary Infos", []))
    chans = _kv(sec.get("Channel Infos", []))
    n = int(common["NumberOfChannels"])
    names, res = [], []
    for i in range(1, n + 1):
        fields = chans[f"Ch{i}"].split(",")
        names.append(fields[0])
        res.append(float(fields[2]) if len(fields) > 2 and fields[2] else 1.0)
    ref = re.search(r"Reference Channel Name\s*=\s*(\S+)", text)
    coords = _kv(sec.get("Coordinates", []))
    return Header(path=Path(path), data_file=common.get("DataFile", ""), marker_file=common.get("MarkerFile", ""),
                  n_channels=n, sampling_interval_us=float(common["SamplingInterval"]),
                  binary_format=binary.get("BinaryFormat", ""), orientation=common.get("DataOrientation", ""),
                  channels=names, resolutions=res, reference_name=ref.group(1) if ref else None,
                  has_coordinates=len(coords) > 0)


def read_markers(path: Path) -> list[Marker]:
    sec = _ini_sections(Path(path).read_text(encoding="utf-8", errors="replace"))
    out = []
    for key, val in _kv(sec.get("Marker Infos", [])).items():
        if not key.startswith("Mk"):
            continue
        f = val.split(",")
        out.append(Marker(kind=f[0], description=f[1].strip() if len(f) > 1 else "",
                          position=int(f[2]), extra=f[5] if len(f) > 5 else ""))
    return out


def stimulus_onset(markers: list[Marker]) -> int | None:
    """Zero-based sample of the first 'S  1' marker (BrainVision positions are one-based)."""
    for m in markers:
        if m.kind == "Stimulus" and m.description.replace(" ", "") == "S1":
            return m.position - 1
    return None


def memmap_eeg(header: Header, eeg_path: Path | None = None) -> np.ndarray:
    """(n_samples, n_channels) float32 view of a multiplexed IEEE_FLOAT_32 file, in file units."""
    if header.binary_format != "IEEE_FLOAT_32" or header.orientation != "MULTIPLEXED":
        raise ValueError(f"{header.path.name}: unsupported format {header.binary_format}/{header.orientation}")
    p = eeg_path or header.path.with_suffix(".eeg")
    n_bytes = p.stat().st_size
    if n_bytes % (4 * header.n_channels):
        raise ValueError(f"{p.name}: size {n_bytes} is not a multiple of {4 * header.n_channels}")
    return np.memmap(p, dtype="<f4", mode="r", shape=(n_bytes // (4 * header.n_channels), header.n_channels))


@dataclass
class Recording:
    subject: str
    condition: str
    part: int
    header: Header
    markers: list[Marker] = field(repr=False)

    @property
    def onset(self) -> int | None:
        return stimulus_onset(self.markers)

    def data(self) -> np.ndarray:
        return memmap_eeg(self.header)


def recording(root: Path, subject: str, condition: str, part: int, suffix: str | None = None) -> Recording:
    stem = f"{subject}_{condition}_{part}" + (f"_{suffix}" if suffix else "")
    hdr = read_header(root / "eeg" / subject / f"{stem}.vhdr")
    return Recording(subject, condition, part, hdr, read_markers(hdr.path.with_suffix(".vmrk")))


def stimulus_files(root: Path, condition: str) -> list[Path]:
    """Stimulus parts in the order stimToCND.m assigns them (directory listing, sorted)."""
    return sorted((root / "stim" / condition).glob("*.mat"))


@dataclass
class Stimulus:
    condition: str
    part: int
    fname: str
    fs: float
    target: np.ndarray = field(repr=False)
    noise: np.ndarray | None = field(default=None, repr=False)
    english: np.ndarray | None = field(default=None, repr=False)
    english_segments: np.ndarray | None = None
    pad: int = 0

    @property
    def is_dutch(self) -> bool:
        return self.condition.endswith("Dutch")

    def aligned_tracks(self) -> tuple[np.ndarray, np.ndarray]:
        """Target and noise tracks on a common time base starting at the audio-file onset,
        reproducing the padding of stimToCND.m."""
        t = np.asarray(self.target, dtype=float)
        if self.noise is None:
            return t, np.zeros_like(t)
        b = np.asarray(self.noise, dtype=float)
        if self.condition in ("lb", "mb", "hb"):
            t = np.pad(t, (self.pad, self.pad))
        elif self.is_dutch:
            b = np.pad(b, (self.pad, 0))
        n = min(t.size, b.size) if self.condition in ("fM", "fW") else max(t.size, b.size)
        t = np.pad(t, (0, max(0, n - t.size)))[:n]
        b = np.pad(b, (0, max(0, n - b.size)))[:n]
        return t, b

    def mixture(self, snr_db: float | None) -> np.ndarray:
        """Presented audio: target (plus embedded English) with the noise track scaled to ``snr_db``
        (RMS over non-silent samples); the stored noise tracks are level-normalised."""
        t, b = self.aligned_tracks()
        if self.english is not None:
            e = np.asarray(self.english, float)
            t = t + np.pad(e, (0, max(0, t.size - e.size)))[:t.size]
        if self.condition in ("fM", "fW"):
            return t + b
        if snr_db is None or not np.any(b):
            return t
        rt, rb = np.sqrt(np.mean(t[t != 0] ** 2)), np.sqrt(np.mean(b[b != 0] ** 2))
        return t + b * rt / rb / 10 ** (snr_db / 20)


def load_stimulus(path: Path, condition: str, part: int) -> Stimulus:
    m = sio.loadmat(path, squeeze_me=True)
    fs = float(m["Fs"])
    if condition.endswith("Dutch"):
        beg = np.atleast_1d(m["idxBeginEn"]).astype(int)
        end = np.atleast_1d(m["idxEndEn"]).astype(int)
        return Stimulus(condition, part, path.name, fs, np.asarray(m["dutch"], float),
                        np.asarray(m["babble"], float) if "babble" in m else None,
                        np.asarray(m["english"], float), np.column_stack([beg, end]),
                        int(m["nRamp"]) if "nRamp" in m else 0)
    if condition in ("fM", "fW"):
        return Stimulus(condition, part, path.name, fs, np.asarray(m["story"], float),
                        np.asarray(m["distractor"], float))
    return Stimulus(condition, part, path.name, fs, np.asarray(m["story"], float),
                    np.asarray(m["babble"], float) if "babble" in m else None,
                    pad=int(m["ntaper"]) if "ntaper" in m else 0)
