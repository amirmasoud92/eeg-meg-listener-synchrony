"""Sung, hummed and spoken versions of the same songs (CantiSong).

Leave-one-song-out spaces, ISC per song and version, and the speech-specific excess L: the sung-minus-hummed difference
in the speech-defined space minus the same difference in the melody-defined space.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.io.wavfile as wavfile
import scipy.signal as ss

from common.settings import dataset_root, load_analysis, load_paths, rng as make_rng
from encoding_models import speech_envelope as tk
from preprocessing import cnd_recordings as cnd, signal_operations as pp
from preprocessing.epoch_rejection import reject
from preprocessing.epochs import Trial
from readers import etard_release as etard
from statistical_tests.inference import summarise
from synchrony.correlated_components import _cov_terms, corrca_from_terms, forward_model, subject_isc

ROOT = dataset_root("cantisong")
VERSIONS = {"speech": "speech", "song": "song", "melody": "melody"}
BANDS = {"1-4": (1.0, 4.0), "0.5-4": (0.5, 4.0), "1-3": (1.0, 3.0)}
FS, START_S, END_TRIM_S, MAX_BAD = 64, 5.0, 2.0, 1 / 3
K = 5


def song_ids() -> list[str]:
    return sorted(p.stem for p in (ROOT / "Stimuli" / "lyrics").glob("*.txt"))


def stim_lengths() -> dict[str, float]:
    t = pd.read_csv(ROOT / "Stimuli" / "sounds" / "stimuli_lengths.csv")
    return dict(zip(t.File.str.replace(".wav", "", regex=False), t.Length))


def load_version(sub: int, version: str):
    e = sio.loadmat(ROOT / "EEG" / version / f"pre_dataSub{sub}.mat", squeeze_me=True, struct_as_record=False)["eeg"]
    return [np.asarray(d, float) for d in np.atleast_1d(e.data)], [str(c.labels).split(" (")[0] for c in np.atleast_1d(e.chanlocs)]


def prepare_one(sub: int, cfg2: dict, seed: int, cache: Path) -> str:
    target = cache / f"sub{sub:02d}.npz"
    if target.exists():
        return target.name
    ids, lengths = song_ids(), stim_lengths()
    items, labels = [], None
    for version in VERSIONS:
        data, labels = load_version(sub, version)
        for k, x in enumerate(data):
            if x.size == 0:
                continue
            dur = min(x.shape[0] / FS, lengths[f"{ids[k]}_{version}"])
            n_ep = int((dur - START_S - END_TRIM_S) // 2)
            if n_ep < 3:
                continue
            items.append((version, ids[k], x[: int(dur * FS)].T * 1e6, n_ep))
    info = cnd.montage_info(labels, FS, "biosemi64")
    a = int(START_S * FS)
    xs = [pp.robust_detrend(x, 3, 3.0) for _, _, x, _ in items]
    bads = pp.bad_channels([x[:, a:a + n * 128] for x, (_, _, _, n) in zip(xs, items)], labels, 5.0)
    op = cnd.spatial_operator(info, bads)
    rank = int((np.linalg.svd(op, compute_uv=False) > 1e-8).sum())
    fit = np.concatenate([(op @ x)[:, a:a + n * 128] for x, (_, _, _, n) in zip(xs, items)], axis=1)
    proj, removed = cnd.ocular_ica(fit, info, min(rank, 40), 4, 3.0, 3, int(make_rng(seed, "canti", sub).integers(2**31)))
    full = proj @ op
    trials = [Trial(subject=sub, condition=v, excerpt=int(i), label=f"{v}_{i}", presentation_index=j, engagement=np.nan,
                    familiarity=np.nan, sfreq=FS, t0=START_S, data=(full @ x)[:, a:a + n * 128])
              for j, (x, (v, i, _, n)) in enumerate(zip(xs, items))]
    res = reject(trials, labels, 2.0, seed=seed, max_bad_fraction=cfg2["max_bad_epoch_fraction"], method=cfg2["epoch_rejection"],
                 z_threshold=cfg2["faster_z"], channel_robust_z=cfg2["channel_robust_z"])
    arrays, meta = {}, []
    for x, t, (v, i, _, n) in zip(xs, trials, items):
        ok = res.epoch_ok[t.key]
        excluded = (~ok).mean() > MAX_BAD
        valid = np.repeat(ok & ~excluded, 128)
        for bname, (lo, hi) in BANDS.items():
            arrays[f"{bname}|{v}|{i}"] = cnd.band_64(full @ x, FS, lo, hi)[:, a:a + n * 128].astype(np.float32)
        arrays[f"valid|{v}|{i}"] = valid
        meta.append({"version": v, "song": i, "n_epochs": n, "reject_fraction": float((~ok).mean()), "excluded": bool(excluded)})
    np.savez(target, **arrays, meta=json.dumps({"bads": bads, "ica": [int(c) for c in removed], "trials": meta}))
    return target.name


def load_all(cache: Path, band: str) -> dict:
    """{(version, song): {subject: (data, valid)}}"""
    out = {}
    for f in sorted(cache.glob("sub*.npz")):
        sub = int(f.stem[3:])
        z = np.load(f)
        for key in z.files:
            if key.startswith(band + "|"):
                _, v, i = key.split("|")
                out.setdefault((v, i), {})[sub] = (z[key].astype(float), z[f"valid|{v}|{i}"])
    return out


def stack(d: dict, subs: list[int]) -> tuple[np.ndarray, np.ndarray]:
    n = min(d[s][0].shape[1] for s in subs)
    return np.stack([d[s][0][:, :n] for s in subs]), np.stack([d[s][1][:n] for s in subs])


def usable(d: dict) -> list[int]:
    return sorted(s for s, (_, v) in d.items() if v.any())


def space(data: dict, version: str, exclude: str, ncomp: int) -> np.ndarray:
    Rw, Rb = 0.0, 0.0
    for (v, i), d in data.items():
        if v != version or i == exclude:
            continue
        subs = usable(d)
        if len(subs) < 3:
            continue
        w, b = _cov_terms(*stack(d, subs))
        Rw, Rb = Rw + w, Rb + b
    return corrca_from_terms(Rw, Rb, 10, 0.5, ncomp)[0]


def per_song(data: dict, ncomp: int) -> pd.DataFrame:
    rows = []
    for s in song_ids():
        if ("song", s) not in data or ("melody", s) not in data:
            continue
        P = sorted(set(usable(data[("song", s)])) & set(usable(data[("melody", s)])))
        if len(P) < 4:
            continue
        Wsp, Wmel = space(data, "speech", s, ncomp), space(data, "melody", s, ncomp)
        isc = {}
        for sp_name, W in (("speech_space", Wsp), ("melody_space", Wmel)):
            for v in ("song", "melody"):
                isc[(sp_name, v)] = subject_isc(*(lambda dv: (dv[0], W, dv[1]))(stack(data[(v, s)], P))).sum(1)
        P3 = sorted(set(P) & set(usable(data.get(("speech", s), {})))) if ("speech", s) in data else []
        spoken = {}
        if len(P3) >= 4:
            for v in ("song", "melody", "speech"):
                spoken[v] = np.nanmean(subject_isc(*(lambda dv: (dv[0], Wsp, dv[1]))(stack(data[(v, s)], P3))).sum(1))
        for j, sub in enumerate(P):
            rows.append({"song": s, "subject": sub,
                         "sp_sung": isc[("speech_space", "song")][j], "sp_hum": isc[("speech_space", "melody")][j],
                         "mel_sung": isc[("melody_space", "song")][j], "mel_hum": isc[("melody_space", "melody")][j],
                         "cap_sung": spoken["song"] / spoken["speech"] if spoken else np.nan,
                         "cap_hum": spoken["melody"] / spoken["speech"] if spoken else np.nan})
    df = pd.DataFrame(rows)
    df["L"] = (df.sp_sung - df.sp_hum) - (df.mel_sung - df.mel_hum)
    return df


def summary(df: pd.DataFrame) -> dict:
    items = df.groupby("song")[["L", "sp_sung", "sp_hum", "mel_sung", "mel_hum", "cap_sung", "cap_hum"]].mean()
    part = df.groupby("subject").L.mean()
    pick = lambda s: {k: s[k] for k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}
    return {"primary_items_L": pick(summarise(items.L.to_numpy(), "greater")),
            "participants_L": pick(summarise(part.to_numpy(), "greater")),
            "items_sung_minus_hummed_speech_space": pick(summarise((items.sp_sung - items.sp_hum).to_numpy(), "greater")),
            "items_sung_minus_hummed_melody_space": pick(summarise((items.mel_sung - items.mel_hum).to_numpy(), "two-sided")),
            "items_capture_ratio_sung_minus_hummed": pick(summarise((items.cap_sung - items.cap_hum).dropna().to_numpy(), "greater")),
            "item_means": items.mean().round(4).to_dict(), "n_songs": int(len(items))}


def envelope_residual(data: dict) -> dict:
    """Remove each participant's cross-validated envelope prediction (lags 0-500 ms) from every trial."""
    env_cfg = etard.load_config().section("envelope")
    envs = {}
    for (v, s) in data:
        fs, audio = wavfile.read(ROOT / "Stimuli" / "sounds" / f"{s}_{v}.wav")
        audio = np.asarray(audio, float)
        audio = audio[:, 0] if audio.ndim > 1 else audio
        e1k = tk.broadband_envelope(audio, fs, env_cfg)
        fr = Fraction(FS, 1000)
        e64 = ss.resample_poly(e1k, fr.numerator, fr.denominator)
        eb = cnd.band_64(e64[None], FS, *BANDS["1-4"])[0]
        envs[(v, s)] = eb[int(START_S * FS):]
    n_lags = 33
    lag = lambda e: np.column_stack([np.r_[np.zeros(k), e[: e.size - k]] for k in range(n_lags)])
    subs = sorted({sub for d in data.values() for sub in d})
    out = {key: {} for key in data}
    for sub in subs:
        keys = [key for key in data if sub in data[key] and data[key][sub][1].any()]
        for key in keys:
            A = np.vstack([lag(envs[k][: data[k][sub][0].shape[1]])[data[k][sub][1]] for k in keys if k != key])
            B = np.vstack([data[k][sub][0][:, data[k][sub][1]].T for k in keys if k != key])
            B = B - B.mean(0)
            G = A.T @ A
            W = np.linalg.solve(G + np.trace(G) / n_lags * np.eye(n_lags), A.T @ B)
            x, v = data[key][sub]
            out[key][sub] = (x - (lag(envs[key][: x.shape[1]]) @ W).T, v)
    return out


def terms_all(data: dict, version: str) -> tuple[np.ndarray, np.ndarray]:
    Rw, Rb = 0.0, 0.0
    for (v, _), d in data.items():
        subs = usable(d)
        if v == version and len(subs) >= 3:
            w, b = _cov_terms(*stack(d, subs))
            Rw, Rb = Rw + w, Rb + b
    return Rw, Rb


def excess_by_component():
    paths, cfg = load_paths(), load_analysis()
    out = paths.results_root
    data = load_all(paths.derived_root / "cantisong", "1-4")
    rows = []
    for s in song_ids():
        if ("song", s) not in data or ("melody", s) not in data:
            continue
        P = sorted(set(usable(data[("song", s)])) & set(usable(data[("melody", s)])))
        if len(P) < 4:
            continue
        W = {"sp": space(data, "speech", s, K), "mel": space(data, "melody", s, K)}
        isc = {(sp, v): subject_isc(*(lambda dv: (dv[0], W[sp], dv[1]))(stack(data[(v, s)], P))).mean(0)
               for sp in W for v in ("song", "melody")}
        for k in range(K):
            rows.append({"song": s, "component": k + 1, "n_participants": len(P),
                         "sp_sung": isc[("sp", "song")][k], "sp_hum": isc[("sp", "melody")][k],
                         "mel_sung": isc[("mel", "song")][k], "mel_hum": isc[("mel", "melody")][k]})
    df = pd.DataFrame(rows)
    df["sp_diff"] = df.sp_sung - df.sp_hum
    df["mel_diff"] = df.mel_sung - df.mel_hum
    df["L"] = df.sp_diff - df.mel_diff
    df.to_csv(out / "song_excess_components.csv", index=False)

    pick = lambda x: {k: v for k, v in summarise(x, "two-sided").items() if k in ("mean", "dz", "dz_ci95", "n_positive")}
    per_comp = {int(k): {c: pick(g[c].to_numpy()) for c in ("sp_diff", "mel_diff", "L")} | {"sp_sung": float(g.sp_sung.mean()), "mel_hum": float(g.mel_hum.mean())}
                for k, g in df.groupby("component")}
    A = {}
    for v in ("speech", "melody"):
        Rw, Rb = terms_all(data, v)
        W, lam = corrca_from_terms(Rw, Rb, 10, 0.5, 3)
        A[v] = forward_model(Rw, W)
        A[v + "_train_isc"] = lam
    topo_r = np.corrcoef(np.vstack([A["speech"].T, A["melody"].T]))[:3, 3:]
    res = {"per_component": per_comp,
           "topography_r_speech_vs_melody_components_1to3": np.round(topo_r, 3).tolist(),
           "training_isc_speech": A["speech_train_isc"].round(4).tolist(), "training_isc_melody": A["melody_train_isc"].round(4).tolist()}
    np.savez(out / "song_excess_topographies.npz", speech=A["speech"], melody=A["melody"])
    (out / "song_excess_components.json").write_text(json.dumps(res, indent=2, default=float))
