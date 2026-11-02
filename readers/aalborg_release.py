"""Readers for the Aalborg song and speech release (Simon et al., 2024): trial structures and channel names."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio

_TARGET = re.compile(r"^\s*(\d+)_([A-Z]{2})\.?\s*$")
_PREFIX = re.compile(r"^([A-Za-z]+?)(z|\d+)$")


def parse_target(label: str) -> tuple[int, str]:
    m = _TARGET.match(str(label))
    if m is None:
        raise ValueError(f"cannot parse stimulus label '{label}'")
    return int(m.group(1)), m.group(2)


@dataclass(frozen=True)
class SubjectInfo:
    subject: int
    gender: str
    age: float
    n_channels: int
    eeg_sfreq: float
    sound_sfreq: float


@dataclass
class PreprocessedTrial:
    subject: int
    file_index: int
    label: str
    excerpt: int
    condition: str
    engagement: float
    familiarity: float
    trial_field: int
    eeg: np.ndarray = field(repr=False)

    @property
    def key(self) -> tuple[int, str, int]:
        return self.subject, self.condition, self.excerpt


@dataclass
class PreprocessedSubject:
    info: SubjectInfo
    trials: list[PreprocessedTrial]

    def by_condition(self, condition: str) -> list[PreprocessedTrial]:
        return [t for t in self.trials if t.condition == condition]


def _scalar(v) -> float:
    arr = np.asarray(v)
    if arr.size != 1:
        raise ValueError(f"expected scalar, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def load_preprocessed(path: Path, subject: int | None = None) -> PreprocessedSubject:
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    data = mat["Data"]
    inf = data.info
    info = SubjectInfo(
        subject=int(inf.subject),
        gender=str(inf.gender),
        age=float(inf.age),
        n_channels=int(inf.nbChannel),
        eeg_sfreq=float(inf.EEG_fs),
        sound_sfreq=float(inf.sound_fs),
    )
    if subject is not None and info.subject != subject:
        raise ValueError(f"{path.name}: info.subject={info.subject}, expected {subject}")
    records = np.atleast_1d(data.data)
    trials = []
    for i, rec in enumerate(records):
        excerpt, cond = parse_target(rec.TargetType)
        trials.append(
            PreprocessedTrial(
                subject=info.subject,
                file_index=i,
                label=str(rec.TargetType),
                excerpt=excerpt,
                condition=cond,
                engagement=_scalar(rec.Engagement),
                familiarity=_scalar(rec.Familiarity),
                trial_field=int(_scalar(rec.Trial)),
                eeg=np.asarray(rec.EEG),
            )
        )
    return PreprocessedSubject(info=info, trials=trials)


@dataclass
class ProcessedTrial:
    participant: int
    presentation_index: int
    label: str
    excerpt: int
    condition: str
    engagement: float
    familiarity: float
    eeg: np.ndarray = field(repr=False)
    envelope: np.ndarray = field(repr=False)


def load_processed(path: Path) -> list[ProcessedTrial]:
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"{path.name}: expected one variable, found {keys}")
    out = []
    for rec in np.atleast_1d(mat[keys[0]]):
        excerpt, cond = parse_target(rec.Audio_Name)
        out.append(
            ProcessedTrial(
                participant=int(_scalar(rec.Participant)),
                presentation_index=int(_scalar(rec.Trial)),
                label=str(rec.Audio_Name),
                excerpt=excerpt,
                condition=cond,
                engagement=_scalar(rec.Engagement),
                familiarity=_scalar(rec.Familiarity),
                eeg=np.asarray(rec.EEG_processed),
                envelope=np.asarray(rec.Audio_Envelope, dtype=float).ravel(),
            )
        )
    return out


def preprocessed_path(directory: Path, subject: int) -> Path:
    return directory / f"S{subject}_preprocessed.mat"


def standardize_name(name: str) -> str:
    """Map release spellings ('Af7', 'Fcz', 'Po3') to 10-05 nomenclature ('AF7', 'FCz', 'PO3')."""
    name = name.strip()
    m = _PREFIX.match(name)
    if m is None:
        raise ValueError(f"unrecognised channel label '{name}'")
    prefix, suffix = m.groups()
    prefix = "Fp" if prefix.lower() == "fp" else prefix.upper()
    return prefix + suffix


def load_channel_names(path: Path) -> list[str]:
    mat = sio.loadmat(path, squeeze_me=True)
    raw = [str(c) for c in mat["ch_list"]]
    names = [standardize_name(c) for c in raw]
    if len(set(names)) != len(names):
        raise ValueError("duplicate channel names after standardisation")
    return names
