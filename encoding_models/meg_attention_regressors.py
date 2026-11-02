"""Stimulus regressors per chapter and stream at 64 Hz for the MEG attention dataset.

Acoustic rows (broadband envelope and onsets, 16-band spectrogram and onsets), word onsets, four controls (impulses
scaled by frequency, function word, duration and peak envelope) and raw surprisal impulses with their onset indices.
"""
from __future__ import annotations

import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss
from scipy.io import wavfile

from common.settings import dataset_root
from encoding_models import speech_envelope as tk
from readers import etard_release as etard

ROOT = Path(__file__).parents[1]
AUDIO = dataset_root("meg_attention") / "Audio"
TAB = ROOT / "results"
OUT = ROOT / "data" / "derived" / "meg_attention" / "features"
FILES = {"LP": {"attended": 1, "ignored": 4}, "HP": {"attended": 3, "ignored": 2}}
FS = 64


def rng(key: str) -> np.random.Generator:
    return np.random.default_rng([7, zlib.crc32(key.encode())])


def wav_path(book: int, k: int) -> Path:
    sub = "attended_speech" if book in (1, 3) else "ignored_speech"
    return AUDIO / sub / f"audiobook{book}_{k}.wav"


def acoustic(path: Path, n: int) -> np.ndarray:
    fs, x = wavfile.read(path)
    x = np.asarray(x, float)
    if x.dtype.kind != "f" or np.abs(x).max() > 1.5:
        x = x / 32768.0
    x = x.mean(1) if x.ndim > 1 else x
    e1k = tk.broadband_envelope(x, fs, etard.load_config().section("envelope"))
    env = ss.resample_poly(e1k, 64, 1000)
    y = ss.resample_poly(x, 16000, fs)
    f, _, Z = ss.stft(y, 16000, nperseg=512, noverlap=512 - 250, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    edges = 700 * (10 ** (np.linspace(np.log10(1 + 100 / 700), np.log10(1 + 7000 / 700), 18)) - 1)
    bands = np.array([P[(f >= lo) & (f < hi)].sum(0) for lo, hi in zip(edges[:-2], edges[2:])]) ** 0.3
    m = min(env.size, bands.shape[1], n)
    A = np.zeros((18, n))
    A[0, :m], A[1:17, :m] = env[:m], bands[:, :m]
    on = np.maximum(np.diff(A, axis=-1, prepend=A[:, :1]), 0)
    return np.vstack([A[:1], on[:1], A[1:17], on[1:17]])


def zs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, np.nanmean(x))
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def impulses(idx: np.ndarray, vals: list[np.ndarray], n: int) -> np.ndarray:
    R = np.zeros((len(vals), n))
    ok = (idx >= 0) & (idx < n)
    for k, v in enumerate(vals):
        np.add.at(R[k], idx[ok], np.asarray(v, float)[ok])
    return R


def chapter_regressors():
    w = pd.read_csv(TAB / "meg_attention_word_features.csv")
    w = w[w.aligned].copy()
    a = pd.read_csv(TAB / "meg_attention_audio.csv")
    for f, books in FILES.items():
        for k in range(1, 6):
            n1000 = int(a[(a.book == f"audiobook{books['attended']}") & (a.chapter == k)].n1000_ceil.iloc[0])
            n = math.ceil(n1000 * 8 / 125)
            arrays = {"n": np.array(n)}
            for s, book in books.items():
                g = w[(w.book == book) & (w.chapter == k)].sort_values("word_index")
                idx = np.round(g.onset.to_numpy() * FS).astype(int)
                arrays[f"{s}_acoustic_raw"] = acoustic(wav_path(book, k), n)
                arrays[f"{s}_word_raw"] = impulses(idx, [np.ones(len(g))], n)
                arrays[f"{s}_controls_raw"] = impulses(idx, [zs(g.zipf), zs(g.function.astype(float)), zs(g.duration), zs(g.peak_env)], n)
                arrays[f"{s}_onset_idx"] = idx
                arrays[f"{s}_surprisal"] = g.surprisal_gerpt2.to_numpy()
            np.savez(OUT / f"{f}_{k}.npz", **arrays)
