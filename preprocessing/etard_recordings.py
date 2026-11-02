"""Etard & Reichenbach (2019) recordings in the format of the network-state analysis.

Per listener and recording session: robust detrending, 0.5-45 Hz Hamming FIR filtering, 250 Hz, bad-channel
interpolation, restoration of the reference site, average reference and one ocular ICA for the whole session. Each story
part becomes a Trial; 2 s epochs overlapping embedded English sentences (plus margins) are marked unavailable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import signal_operations as pp
from preprocessing.epochs import Trial
from readers import etard_release as etard

def fir_order(sfreq: float, transition_hz: float) -> int:
    m = int(np.ceil(3.3 / (transition_hz / sfreq)))
    return m + (m % 2)


def filter_recording(x: np.ndarray, onset: int, start: int, stop: int, cfg: dict) -> np.ndarray:
    """Detrend, low-pass at 1 kHz, decimate in phase with ``onset``, high-pass; return samples
    [start, stop) of the output grid (indices relative to onset, in output samples)."""
    x = pp.robust_detrend(x, cfg["detrend_order"], cfg["detrend_threshold_sd"])
    lp = cfg["lowpass"]
    b = pp.fir_kernel(lp["edge_hz"] + lp["transition_hz"] / 2, 1000.0, fir_order(1000.0, lp["transition_hz"]), False)
    y = pp.fir_filter(x, b)
    step = int(1000 // cfg["sfreq"])
    y = y[:, onset % step::step]
    hp = cfg["highpass"]
    b = pp.fir_kernel(hp["edge_hz"] - hp["transition_hz"] / 2, float(cfg["sfreq"]),
                      fir_order(float(cfg["sfreq"]), hp["transition_hz"]), True)
    y = pp.fir_filter(y, b)
    o = onset // step
    return y[:, o + start: o + stop]


@dataclass
class PartInfo:
    subject: str
    session: str
    condition: str
    part: int
    order: int
    duration_s: float
    english_s: np.ndarray | None


def session_parts(root: Path, sid: str, session: str, conditions: list[str]) -> list[PartInfo]:
    out, k = [], 0
    for cond in conditions:
        for part, f in enumerate(etard.stimulus_files(root, cond), start=1):
            st = etard.load_stimulus(f, cond, part)
            t, _ = st.aligned_tracks()
            seg = None if st.english_segments is None else st.english_segments / st.fs
            out.append(PartInfo(sid, session, cond, part, k, t.size / st.fs, seg))
            k += 1
    return out


def available_epochs(n_epochs: int, epoch_s: float, window_start_s: float, english_s: np.ndarray | None,
                     margin_s: tuple[float, float]) -> np.ndarray:
    """2 s epochs of the analysis window that do not overlap embedded English (plus margins)."""
    ok = np.ones(n_epochs, dtype=bool)
    if english_s is None:
        return ok
    starts = window_start_s + np.arange(n_epochs) * epoch_s
    for beg, end in np.atleast_2d(english_s):
        a, b = beg - margin_s[0], end + margin_s[1]
        ok &= ~((starts < b) & (starts + epoch_s > a))
    return ok


def preprocess_session(root: Path, sid: str, session: str, conditions: list[str], cfg: dict, epoch_s: float,
                       out: Path, seed: int) -> dict:
    """Preprocess one recording session and save every part as a float32 array plus metadata."""
    target = out / f"{sid}_{session}.json"
    if target.exists():
        return json.loads(target.read_text())
    sf = int(cfg["sfreq"])
    parts = session_parts(root, sid, session, conditions)
    data, meta_parts, ch_names = [], [], None
    for p in parts:
        rec, x = pp.raw_part(root, sid, p.condition, p.part)
        ch_names = [c for c in rec.header.channels if c != "Sound"]
        n_win = int(np.floor((p.duration_s - cfg["window_start_s"] - cfg["window_end_trim_s"]) / epoch_s))
        start = int(round((cfg["window_start_s"] - cfg["pad_s"]) * sf))
        stop = int(round(cfg["window_start_s"] * sf)) + int(round(n_win * epoch_s * sf))
        avail_after = (x.shape[1] - rec.onset) // int(1000 // sf)
        if stop > avail_after:
            n_win = int(np.floor((avail_after - int(round(cfg["window_start_s"] * sf))) / (epoch_s * sf)))
            stop = int(round(cfg["window_start_s"] * sf)) + int(round(n_win * epoch_s * sf))
        data.append(filter_recording(x, rec.onset, start, stop, cfg))
        meta_parts.append({"condition": p.condition, "part": p.part, "order": p.order, "n_epochs": n_win,
                           "available": available_epochs(n_win, epoch_s, cfg["window_start_s"], p.english_s,
                                                         tuple(cfg["english_margin_s"])).tolist()})
    pad = int(round(cfg["pad_s"] * sf))
    bads = pp.bad_channels([d[:, pad:] for d in data], ch_names, cfg["bad_channel_robust_z"])
    op, names = pp.spatial_operator(ch_names, bads, float(sf))
    rank = pp.rank_after(op)
    ic = cfg["ica"]
    fit_step = sf // int(ic["fit_sfreq"])
    b = pp.fir_kernel(ic["fit_highpass_hz"] / 2, float(sf), fir_order(float(sf), ic["fit_highpass_hz"]), True)
    fit = np.concatenate([pp.fir_filter(op @ d[:, pad:], b)[:, ::fit_step] for d in data], axis=1)
    proj, excluded, scores = pp.fit_ica(fit, names, float(ic["fit_sfreq"]), rank, ic, seed)
    full = (proj @ op).astype(np.float64)
    for p, d, m in zip(parts, data, meta_parts):
        np.save(out / f"{sid}_{p.condition}_{p.part}.npy", (full @ d).astype(np.float32))
    meta = {"subject": sid, "session": session, "channels": names, "bads": bads, "rank": rank,
            "ica_excluded": [int(i) for i in excluded], "ica_max_score": float(np.max(scores)),
            "sfreq": sf, "pad": pad, "window_start_s": cfg["window_start_s"], "parts": meta_parts}
    target.write_text(json.dumps(meta))
    return meta


class EtardDataset:
    """Trials of the preprocessed recordings, mirroring ``aalborg_trials.CleanDataset``."""

    def __init__(self, cache: Path, participants: list[str]):
        self.cache = cache
        self.participants = participants
        self.meta = {}
        for sid in participants:
            for f in sorted(cache.glob(f"{sid}_*.json")):
                m = json.loads(f.read_text())
                self.meta[(sid, m["session"])] = m

    @property
    def channels(self) -> list[str]:
        return next(iter(self.meta.values()))["channels"]

    def table(self) -> pd.DataFrame:
        rows = []
        for (sid, session), m in self.meta.items():
            for p in m["parts"]:
                rows.append({"subject": sid, "session": session, "condition": p["condition"], "part": p["part"],
                             "order": p["order"], "key": f"{sid}_{p['condition']}_{p['part']}",
                             "n_epochs_2s": p["n_epochs"], "n_available_2s": int(np.sum(p["available"]))})
        return pd.DataFrame(rows)

    def trials(self, sid: str, session: str | None = None, with_pad: bool = False) -> list[Trial]:
        out = []
        for (s, sess), m in self.meta.items():
            if s != sid or (session is not None and sess != session):
                continue
            for p in m["parts"]:
                x = np.load(self.cache / f"{sid}_{p['condition']}_{p['part']}.npy").astype(np.float64)
                if not with_pad:
                    x = x[:, m["pad"]:]
                out.append(Trial(subject=sid, condition=p["condition"], excerpt=p["part"], label=f"{p['condition']}_{p['part']}",
                                 presentation_index=p["order"], engagement=np.nan, familiarity=np.nan,
                                 sfreq=float(m["sfreq"]), t0=m["window_start_s"] - (m["pad"] / m["sfreq"] if with_pad else 0.0),
                                 data=x))
        return out

    def available(self, sid: str) -> dict[str, np.ndarray]:
        out = {}
        for (s, _), m in self.meta.items():
            if s == sid:
                for p in m["parts"]:
                    out[f"S{sid}_{p['condition']}{p['part']}"] = np.asarray(p["available"], bool)
        return out
