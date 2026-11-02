"""EEG preprocessing for Continuous-event Neural Data releases (Di Liberto et al., CNSP).

Per listener: robust polynomial detrending, bad channels (log-SD robust z > 5) interpolated by spherical splines,
average reference, one ocular extended-Infomax ICA (at most 40 components; the most anterior channels serve as ocular
proxies), FASTER plus robust channel epoch rejection on 2 s epochs, and the analysis band on a 64 Hz grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mne
import numpy as np
import scipy.io as sio

from preprocessing import signal_operations as pp
from preprocessing.epoch_rejection import reject
from preprocessing.epochs import Trial

def load_subject(path) -> dict:
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    e = m["eeg"]
    data = [np.asarray(d, float) for d in np.atleast_1d(e.data)]
    labels = [str(c.labels) for c in np.atleast_1d(e.chanlocs)]
    return {"fs": float(e.fs), "runs": data, "labels": labels,
            "order": np.atleast_1d(getattr(e, "origTrialPosition", np.arange(1, len(data) + 1))).tolist()}


def load_stim(path) -> dict:
    s = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["stim"]
    data = np.atleast_2d(s.data)
    return {"fs": float(s.fs), "names": list(np.atleast_1d(s.names)), "envelope": [np.asarray(v, float) for v in data[0]]}


def montage_info(labels: list[str], sfreq: float, montage: str) -> mne.Info:
    info = mne.create_info(labels, sfreq, "eeg")
    info.set_montage(mne.channels.make_standard_montage(montage), match_case=False)
    return info


def spatial_operator(info: mne.Info, bads: list[str]) -> np.ndarray:
    """Interpolation of ``bads`` followed by average reference, as a channels x channels matrix."""
    n = len(info.ch_names)
    raw = mne.io.RawArray(np.eye(n), info.copy(), verbose="error")
    raw.info["bads"] = list(bads)
    if bads:
        raw.interpolate_bads(reset_bads=True, verbose="error")
    op = raw.get_data()
    return op - op.mean(axis=0, keepdims=True)


def ocular_ica(x: np.ndarray, info: mne.Info, rank: int, n_proxies: int, z: float, max_components: int,
               seed: int) -> tuple[np.ndarray, list[int]]:
    """Linear projection removing ocular ICA components; proxies are the most anterior channels."""
    pos = np.array([info["chs"][i]["loc"][:3] for i in range(len(info.ch_names))])
    proxies = [info.ch_names[i] for i in np.argsort(-pos[:, 1])[:n_proxies]]
    raw = mne.io.RawArray(x * 1e-6, info.copy(), verbose="error")
    ica = mne.preprocessing.ICA(n_components=rank, method="infomax", fit_params={"extended": True},
                                random_state=seed, max_iter="auto", verbose="error")
    ica.fit(raw, verbose="error")
    idx, scores = ica.find_bads_eog(raw, ch_name=proxies, threshold=z, measure="zscore", verbose="error")
    score = np.max(np.abs(np.atleast_2d(scores)), axis=0)
    idx = sorted(idx, key=lambda k: -score[k])[:max_components]
    ica.exclude = list(idx)
    n = len(info.ch_names)
    probe = mne.io.RawArray(np.hstack([np.eye(n), np.zeros((n, 1))]) * 1e-6, info.copy(), verbose="error")
    out = ica.apply(probe, verbose="error").get_data()
    return (out[:, :n] - out[:, n:]) / 1e-6, list(idx)


def fir_order(sfreq: float, transition_hz: float) -> int:
    m = int(np.ceil(3.3 / (transition_hz / sfreq)))
    return m + (m % 2)


def band_64(x: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    """1 Hz-transition Hamming band-pass on a 64 Hz grid (anti-alias low-pass before decimation)."""
    step = int(round(sfreq / 64.0))
    y = x
    if step > 1:
        y = pp.fir_filter(y, pp.fir_kernel(25.0, sfreq, fir_order(sfreq, 1.0), False))[:, ::step]
    y = pp.fir_filter(y, pp.fir_kernel(hi, 64.0, fir_order(64.0, 1.0), False))
    return pp.fir_filter(y, pp.fir_kernel(max(lo - 0.5, 0.25), 64.0, fir_order(64.0, 1.0), True))


@dataclass
class Prepared:
    subject: str
    runs: list[np.ndarray] = field(repr=False)     # delta band, channels x samples at 64 Hz (analysis window)
    valid: list[np.ndarray] = field(repr=False)    # per-sample validity at 64 Hz
    broad: list[np.ndarray] = field(repr=False)    # cleaned 1-40 Hz data at the native rate (analysis window)
    bads: list[str] = field(default_factory=list)
    ica_removed: list[int] = field(default_factory=list)
    reject_fraction: float = 0.0


def prepare_subject(sub: dict, durations_s: list[float], montage: str, band: tuple[float, float], start_s: float,
                    end_trim_s: float, seed: int, cfg2: dict, name: str) -> Prepared:
    fs = sub["fs"]
    info = montage_info(sub["labels"], fs, montage)
    hp = pp.fir_kernel(0.5, fs, fir_order(fs, 1.0), True)
    lp = pp.fir_kernel(40.0, fs, fir_order(fs, 5.0), False)
    runs, wins = [], []
    for x, dur in zip(sub["runs"], durations_s):
        n = min(int(round(dur * fs)), x.shape[0])
        y = pp.robust_detrend(x[:n].T, 3, 3.0)
        runs.append(y)
        wins.append((int(round(start_s * fs)), int(round((dur - end_trim_s) * fs))))
    broad_raw = [pp.fir_filter(pp.fir_filter(y, hp), lp) for y in runs]
    bads = pp.bad_channels([b[:, a:e] for b, (a, e) in zip(broad_raw, wins)], sub["labels"], 5.0)
    op = spatial_operator(info, bads)
    rank = int((np.linalg.svd(op, compute_uv=False) > 1e-8).sum())
    step = int(round(fs / 64.0))
    fit = np.concatenate([(op @ b[:, a:e])[:, ::max(step, 1)] for b, (a, e) in zip(broad_raw, wins)], axis=1)
    info64 = montage_info(sub["labels"], fs / max(step, 1), montage)
    proj, removed = ocular_ica(fit, info64, min(rank, 40), 4, 3.0, 3, seed)
    full = proj @ op
    broad, delta, valid = [], [], []
    trials = []
    for k, (y, (a, e)) in enumerate(zip(runs, wins)):
        b = (full @ broad_raw[k])[:, a:e]
        n_ep = b.shape[1] // int(2 * fs)
        b = b[:, : n_ep * int(2 * fs)]
        broad.append(b)
        trials.append(Trial(subject=name, condition="run", excerpt=k, label=str(k), presentation_index=k,
                            engagement=np.nan, familiarity=np.nan, sfreq=fs, t0=start_s, data=b))
        d = band_64(full @ y, fs, *band)
        a64 = int(round(start_s * 64))
        delta.append(d[:, a64: a64 + n_ep * 128].astype(np.float32))
    res = reject(trials, sub["labels"], 2.0, seed=seed, max_bad_fraction=cfg2["max_bad_epoch_fraction"],
                 method=cfg2["epoch_rejection"], z_threshold=cfg2["faster_z"], channel_robust_z=cfg2["channel_robust_z"])
    n_bad, n_all = 0, 0
    for t in trials:
        ok = res.epoch_ok[t.key]
        valid.append(np.repeat(ok, 128)[: delta[t.excerpt].shape[1]])
        n_bad += int((~ok).sum())
        n_all += ok.size
    return Prepared(name, delta, valid, broad, bads, removed, n_bad / max(n_all, 1))
