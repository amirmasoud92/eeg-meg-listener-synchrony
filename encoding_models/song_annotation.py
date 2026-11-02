"""Lyric annotation of the songs: the offset between the released annotation and the audio."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from common.settings import load_paths
from synchrony import song_versions as c

HOP, MAX_LAG_S, FLAG_S = 0.01, 5.0, 0.5


def phones(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) >= 3:
            try:
                out.append((float(p[0]), float(p[1]), p[2]))
            except ValueError:
                continue
    return out


def mask(ph, n: int) -> np.ndarray:
    m = np.zeros(n)
    for a, b, lab in ph:
        if lab.lower() not in ("sil", "sp", "pau", "<p:>", ""):
            m[int(a / HOP):int(b / HOP)] = 1.0
    return m


def rms_mask(wav: Path) -> np.ndarray:
    sr, y = wavfile.read(wav)
    y = y.astype(float)
    y = y.mean(1) if y.ndim > 1 else y
    h = int(sr * HOP)
    r = np.sqrt(np.array([np.mean(y[i:i + h] ** 2) for i in range(0, len(y) - h, h)]))
    return (r > np.percentile(r, 30)).astype(float)


def best_lag(ann: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    n = min(len(ann), len(ref))
    a, r = ann[:n] - ann[:n].mean(), ref[:n] - ref[:n].mean()
    L = int(MAX_LAG_S / HOP)
    cc = {}
    for lag in range(-L, L + 1):
        x, y = (a[:n - lag], r[lag:]) if lag >= 0 else (a[-lag:], r[:n + lag])
        cc[lag] = float(np.corrcoef(x, y)[0, 1])
    k = max(cc, key=cc.get)
    return k * HOP, cc[k], cc[0]


def annotation_offsets():
    paths = load_paths()
    stim = c.ROOT / "Stimuli"
    f0_dir = paths.derived_root / "cantisong_notes_f0"
    out = {}
    for s in c.song_ids():
        semi = np.load(f0_dir / f"{s}_f0_semitones_10ms.npy")
        sung = best_lag(mask(phones(stim / "ph_annotations" / f"{s}_song.txt"), len(semi)), np.isfinite(semi).astype(float))
        ref = rms_mask(stim / "sounds" / f"{s}_speech.wav")
        spoken = best_lag(mask(phones(stim / "ph_annotations" / f"{s}_speech.txt"), len(ref)), ref)
        out[s] = {"song": {"offset_s": sung[0], "r_best": sung[1], "r_zero": sung[2], "flag": abs(sung[0]) > FLAG_S},
                  "speech": {"offset_s": spoken[0], "r_best": spoken[1], "r_zero": spoken[2], "flag": abs(spoken[0]) > FLAG_S}}
    target = paths.derived_root / "cantisong_words" / "annotation_offsets.json"
    target.write_text(json.dumps(out, indent=2))
    (paths.results_root / "cantisong_annotation_offsets.json").write_text(json.dumps(out, indent=2))
