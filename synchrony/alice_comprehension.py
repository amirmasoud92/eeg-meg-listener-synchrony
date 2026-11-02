"""Comprehension and synchrony across listeners in Alice.

EEG preparation, leave-one-out synchrony per listener, the word-level encoding gain per listener, and Spearman
permutation tests against comprehension scores.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import scipy.signal as ss
from joblib import Parallel, delayed
from scipy import stats

from common.settings import dataset_root, rng as make_rng
from encoding_models import song_regressors as g, speech_envelope as tk
from preprocessing import cnd_recordings as cnd, signal_operations as pp
from preprocessing.epoch_rejection import reject
from preprocessing.epochs import Trial
from readers import etard_release as etard
from statistical_tests.inference import summarise
from synchrony.correlated_components import _cov_terms, corrca_from_terms, subject_isc

ROOT = dataset_root("alice")
FS = 128.0
SEGMENTS = range(1, 13)
START_S, END_TRIM_S, MAX_BAD, MIN_SEGMENTS = 1.0, 1.0, 1 / 3, 8
SURPRISAL_COLS = {"gpt2": "gpt2_surprisal_bits", "rnn": "RNN", "ngram": "NGRAM", "cfg": "CFG"}


def wav_path(seg: int) -> Path:
    return ROOT / "audio" / f"DownTheRabbitHoleFinal_SoundFile{seg}.wav"


def seg_durations() -> dict[int, float]:
    out = {}
    for s in SEGMENTS:
        fs, x = wavfile.read(wav_path(s))
        out[s] = len(x) / fs
    return out


def load_raw(sub: str) -> tuple[np.ndarray, list[str], np.ndarray, dict[int, int]]:
    """EEG (channels x samples, µV, 128 Hz), channel labels, VEOG (µV), segment onset samples at 128 Hz."""
    raw = mne.io.read_raw_brainvision(ROOT / f"{sub}.vhdr", preload=True, verbose="error")
    eeg = [c for c in raw.ch_names if c.isdigit()]
    X = raw.get_data(picks=eeg) * 1e6
    veog = raw.get_data(picks=["VEOG"])[0] * 1e6
    fr = Fraction(int(FS), int(raw.info["sfreq"]))
    X = ss.resample_poly(X, fr.numerator, fr.denominator, axis=1)
    veog = ss.resample_poly(veog, fr.numerator, fr.denominator)
    ann = mne.events_from_annotations(raw, verbose="error")
    events, ids = ann
    onsets = {}
    import re
    for code, val in ids.items():
        m = re.search(r"(\d+)\s*$", code.split("/")[-1])      # "1" (S01-S20) or "S  1" (S21-S49)
        tag = m.group(1) if m else ""
        if tag.isdigit() and int(tag) in SEGMENTS:
            samp = events[events[:, 2] == val, 0]
            if samp.size:
                onsets[int(tag)] = int(round(samp[0] * FS / raw.info["sfreq"]))
    return X, eeg, veog, onsets


def montage_info(labels: list[str], axis_flip: dict) -> mne.Info:
    """Head-frame montage from the supplied sfp (cm on a sphere); anterior axis oriented to +y."""
    rows = [ln.split() for ln in (ROOT / "easycapM10-acti61_elec.sfp").read_text().splitlines() if ln.strip()]
    pos = {r[0]: np.array([float(v) for v in r[1:4]]) for r in rows}
    ant, sign = axis_flip["anterior_axis"], axis_flip["anterior_sign"]
    lat = 1 - ant if ant in (0, 1) else 0
    ch_pos = {}
    for c in labels:
        p = pos[c]
        ch_pos[c] = np.array([p[lat], sign * p[ant], p[2]]) / 100.0
    info = mne.create_info(labels, FS, "eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head"))
    return info


def anterior_axis(subs: list[str]) -> dict:
    """Axis (x or y of the sfp) and sign along which channels correlate most with VEOG (blinks, 1-4 Hz)."""
    rows = [ln.split() for ln in (ROOT / "easycapM10-acti61_elec.sfp").read_text().splitlines() if ln.strip()]
    pos = {r[0]: np.array([float(v) for v in r[1:4]]) for r in rows}
    rs = []
    for s in subs:
        X, labels, veog, _ = load_raw(s)
        Xf = cnd.band_64(X - X.mean(0, keepdims=True), FS, 1.0, 4.0)
        vf = cnd.band_64(veog[None], FS, 1.0, 4.0)[0]
        rs.append({c: abs(np.corrcoef(Xf[i], vf)[0, 1]) for i, c in enumerate(labels)})
    r = pd.DataFrame(rs).mean()
    top = r.sort_values(ascending=False).index[:6]
    centroid = np.mean([pos[c] for c in top], axis=0)                    # blinks dominate the frontal channels
    ant = int(np.argmax(np.abs(centroid[:2])))
    best = {0: float(centroid[0]), 1: float(centroid[1])}
    return {"anterior_axis": ant, "anterior_sign": int(np.sign(centroid[ant])), "top6_centroid_xy_cm": best,
            "top_veog_channels": r.sort_values(ascending=False).index[:6].tolist()}


def prepare_one(sub: str, axis: dict, cfg2: dict, seed: int, cache: Path) -> str:
    target = cache / f"{sub}.npz"
    if target.exists():
        return target.name
    X, labels, veog, onsets = load_raw(sub)
    dur = seg_durations()
    info = montage_info(labels, axis)
    hp = pp.fir_kernel(0.5, FS, cnd.fir_order(FS, 1.0), True)
    lp = pp.fir_kernel(40.0, FS, cnd.fir_order(FS, 5.0), False)
    items = []
    for s in SEGMENTS:
        if s not in onsets:
            continue
        a = onsets[s]
        n = int(dur[s] * FS)
        if a + n > X.shape[1]:
            continue
        items.append((s, X[:, a:a + n]))
    ys = [pp.robust_detrend(x, 3, 3.0) for _, x in items]
    broad_raw = [pp.fir_filter(pp.fir_filter(y, hp), lp) for y in ys]
    a0 = int(START_S * FS)
    n_ep = [int((x.shape[1] / FS - START_S - END_TRIM_S) // 2) for _, x in items]
    wins = [(a0, a0 + n * int(2 * FS)) for n in n_ep]
    bads = pp.bad_channels([bb[:, s:e] for bb, (s, e) in zip(broad_raw, wins)], labels, 5.0)
    op = cnd.spatial_operator(info, bads)
    rank = int((np.linalg.svd(op, compute_uv=False) > 1e-8).sum())
    fit = np.concatenate([(op @ bb[:, s:e])[:, ::2] for bb, (s, e) in zip(broad_raw, wins)], axis=1)
    info64 = info.copy()
    with info64._unlock():
        info64["sfreq"] = 64.0
    proj, removed = cnd.ocular_ica(fit, info64, min(rank, 40), 4, 3.0, 3, int(make_rng(seed, "alice", sub).integers(2**31)))
    full = proj @ op
    trials = [Trial(subject=sub, condition="seg", excerpt=s, label=f"seg{s}", presentation_index=j, engagement=np.nan,
                    familiarity=np.nan, sfreq=FS, t0=START_S, data=(full @ bb)[:, s0:e0])
              for j, ((s, _), bb, (s0, e0)) in enumerate(zip(items, broad_raw, wins))]
    res = reject(trials, labels, 2.0, seed=seed, max_bad_fraction=cfg2["max_bad_epoch_fraction"], method=cfg2["epoch_rejection"],
                 z_threshold=cfg2["faster_z"], channel_robust_z=cfg2["channel_robust_z"])
    arrays, meta = {}, []
    a64 = int(START_S * 64)
    for (s, _), y, t, n in zip(items, ys, trials, n_ep):
        ok = res.epoch_ok[t.key]
        excluded = bool((~ok).mean() > MAX_BAD)
        arrays[f"valid|{s}"] = np.repeat(ok & ~excluded, 128)
        for bname, (lo, hi) in (("1-4", (1.0, 4.0)), ("1-8", (1.0, 8.0))):
            arrays[f"{bname}|{s}"] = cnd.band_64(full @ y, FS, lo, hi)[:, a64:a64 + n * 128].astype(np.float32)
        meta.append({"segment": s, "reject_fraction": float((~ok).mean()), "excluded": excluded})
    np.savez(target, **arrays, meta=json.dumps({"bads": bads, "ica": [int(x) for x in removed], "trials": meta,
                                                 "labels": labels}))
    return target.name


def acoustic(seg: int) -> np.ndarray:
    fs, x = wavfile.read(wav_path(seg))
    x = np.asarray(x, float)
    x = x[:, 0] if x.ndim > 1 else x
    e1k = tk.broadband_envelope(x, fs, etard.load_config().section("envelope"))
    env = ss.resample_poly(e1k, 64, 1000)
    y = ss.resample_poly(x, 16000, fs)
    f, _, Z = ss.stft(y, 16000, nperseg=512, noverlap=512 - 250, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    edges = 700 * (10 ** (np.linspace(np.log10(1 + 100 / 700), np.log10(1 + 7000 / 700), 18)) - 1)
    bands = np.array([P[(f >= lo) & (f < hi)].sum(0) for lo, hi in zip(edges[:-2], edges[2:])]) ** 0.3
    n = min(env.size, bands.shape[1])
    on = lambda a: np.maximum(np.diff(a, axis=-1, prepend=a[..., :1]), 0)
    return np.vstack([env[:n], on(env[:n]), bands[:, :n], on(bands[:, :n])])


def seg_stim(seg: int, words: pd.DataFrame, col: str) -> dict:
    A = acoustic(seg)
    n = A.shape[1]
    w = words[words.Segment == seg]
    idx = (w.onset.to_numpy() * 64).round().astype(int)
    keep = (idx >= 0) & (idx < n)
    idx, s = idx[keep], w[col].to_numpy()[keep]
    on, su = np.zeros(n), np.zeros(n)
    np.add.at(on, idx, 1.0)
    np.add.at(su, idx, s)
    return {"acoustic": g.bz(A), "word": g.bz(on[None]), "surprisal": g.bz(su[None]), "word_idx": idx.tolist(), "raw_surprisal": su}


def listener_delta_r(sub: str, data: dict, stims: dict, seed: int) -> dict:
    segs = [s for s in SEGMENTS if s in data and data[s][1].any()]
    if len(segs) < MIN_SEGMENTS:
        return {}
    a = int(START_S * 64)
    rng = np.random.default_rng([seed, int(sub[1:]), 8])
    out = {"listener": sub}
    for name in SURPRISAL_COLS:
        blocks, masks = [], []
        for s in segs:
            st = stims[(s, name)]
            Y, v = data[s]
            T = min(Y.shape[1], st["acoustic"].shape[1] - a)
            X = {k: g.lagged(st[k][:, a:a + T], T) for k in ("acoustic", "word", "surprisal")}
            m = np.zeros(T, bool)
            for i in st["word_idx"]:
                lo = i - a
                if lo + g.WORD_WIN > 0 and lo < T:
                    m[max(lo, 0): min(lo + g.WORD_WIN, T)] = True
            blocks.append({"X": X, "Y": Y[:, :T], "v": v[:T], "st": st, "a": a, "T": T})
            masks.append(m & v[:T])
        _, lam = g.fit_predict(blocks, ["acoustic", "word"], None)
        r_full = g.r_mean(blocks, g.fit_predict(blocks, ["acoustic", "word", "surprisal"], lam)[0], masks)
        r_sh = []
        for _ in range(g.N_SHUF):
            for bl in blocks:
                st = bl["st"]
                raw = np.zeros_like(st["raw_surprisal"])
                raw[st["word_idx"]] = rng.permutation(st["raw_surprisal"][st["word_idx"]])
                bl["X"]["surprisal_shuf"] = g.lagged(g.bz(raw[None])[:, bl["a"]:bl["a"] + bl["T"]], bl["T"])
            r_sh.append(g.r_mean(blocks, g.fit_predict(blocks, ["acoustic", "word", "surprisal_shuf"], lam)[0], masks))
        out[f"delta_r_{name}"] = r_full - float(np.mean(r_sh))
    return out


def load_all(cache: Path, band: str) -> tuple[dict, dict]:
    """{listener: {segment: (data, valid)}} for listeners with >= MIN_SEGMENTS usable segments; QC per listener."""
    data, qc = {}, {}
    for f in sorted(cache.glob("S*.npz")):
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        segs = {s: (z[f"{band}|{s}"].astype(float), z[f"valid|{s}"].astype(bool)) for s in SEGMENTS
                if f"valid|{s}" in z.files and z[f"valid|{s}"].any()}
        qc[f.stem] = {"usable_segments": len(segs), "reject_fraction": float(np.mean([t["reject_fraction"] for t in m["trials"]])),
                      "bads": len(m["bads"]), "ica": len(m["ica"])}
        if len(segs) >= MIN_SEGMENTS:
            data[f.stem] = segs
    return data, qc


def listener_isc(data: dict) -> pd.Series:
    subs = sorted(data)
    vals = {s: [] for s in subs}
    per_seg = {}
    for seg in SEGMENTS:
        have = [s for s in subs if seg in data[s]]
        if len(have) < 4:
            continue
        n = min(data[s][seg][0].shape[1] for s in have)
        per_seg[seg] = (have, np.stack([data[s][seg][0][:, :n] for s in have]), np.stack([data[s][seg][1][:n] for s in have]))
    terms = {seg: _cov_terms(x, v) for seg, (_, x, v) in per_seg.items()}
    for seg, (have, x, v) in per_seg.items():
        W = corrca_from_terms(sum(t[0] for q, t in terms.items() if q != seg), sum(t[1] for q, t in terms.items() if q != seg),
                              len(subs), 0.5, 3)[0]
        isc = subject_isc(x, W, v).sum(1)
        for s, val in zip(have, isc):
            vals[s].append(val)
    return pd.Series({s: float(np.nanmean(v)) for s, v in vals.items() if v})


def scores() -> pd.Series:
    out = {}
    for ln in (ROOT / "comprehension-scores.txt").read_text().splitlines()[1:]:
        parts = ln.split()
        if parts and parts[0].startswith("S") and len(parts) > 1 and "/" in parts[1]:
            a, b = parts[1].split("/")
            out[parts[0]] = int(a) / int(b)
    return pd.Series(out)


def spearman_perm(x, y, n=100000, seed=0) -> tuple[float, float]:
    r = stats.spearmanr(x, y).correlation
    g_ = np.random.default_rng(seed)
    ry = stats.rankdata(y)
    rx = stats.rankdata(x)
    null = np.array([np.corrcoef(rx, g_.permutation(ry))[0, 1] for _ in range(n)])
    return float(r), float((1 + (null >= r).sum()) / (1 + n))


def comprehension_test(paths, cache: Path, seed: int, n_jobs: int) -> tuple[dict, pd.DataFrame]:
    words = pd.read_csv(paths.derived_root / "alice" / "words_gpt2.csv")
    stims = {(s, name): seg_stim(s, words, col) for s in SEGMENTS for name, col in SURPRISAL_COLS.items()}
    data8, qc = load_all(cache, "1-8")
    rows = Parallel(n_jobs=n_jobs)(delayed(listener_delta_r)(sub, data8[sub], stims, seed) for sub in sorted(data8))
    df = pd.DataFrame([r for r in rows if r]).set_index("listener")
    data4, _ = load_all(cache, "1-4")
    df["isc"] = listener_isc(data4)
    sc = scores()
    df["comprehension"] = sc.reindex(df.index)
    df["reject_fraction"] = pd.Series({k: v["reject_fraction"] for k, v in qc.items()}).reindex(df.index)
    use = set(str(x).replace(".mat", "") for x in __import__("scipy.io", fromlist=["loadmat"]).loadmat(
        ROOT / "datasets.mat", squeeze_me=True)["use"])
    df["authors_use"] = df.index.isin(use)
    pick = lambda x: {k: v for k, v in summarise(np.asarray(x, float), "greater").items()
                      if k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}
    res = {"n_prepared": len(qc), "n_included": int(len(df)), "qc": qc}
    res["delta_r_gpt2"] = pick(df.delta_r_gpt2)
    for name in ("rnn", "ngram", "cfg"):
        res[f"sec2_delta_r_{name}"] = pick(df[f"delta_r_{name}"])
    d = df.dropna(subset=["comprehension"])
    res["rho_comprehension_delta_r"], res["p_delta_r_one_sided"] = spearman_perm(d.comprehension, d.delta_r_gpt2, seed=seed)
    r3 = stats.spearmanr(d.comprehension, d.isc)
    res["rho_comprehension_isc"], res["p_isc_two_sided"] = float(r3.correlation), float(r3.pvalue)
    gb = np.random.default_rng(seed)
    diffs = []
    for _ in range(10000):
        ix = gb.integers(0, len(d), len(d))
        b = d.iloc[ix]
        diffs.append(stats.spearmanr(b.comprehension, b.delta_r_gpt2).correlation - stats.spearmanr(b.comprehension, b.isc).correlation)
    diffs = np.array(diffs)
    obs = res["rho_comprehension_delta_r"] - res["rho_comprehension_isc"]
    res["difference"] = {"observed": obs, "ci95": [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))],
                             "p_one_sided": float((np.sum(diffs <= 0) + 1) / (len(diffs) + 1))}

    def partial(x, y, z):
        rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
        ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
        ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
        return float(np.corrcoef(ex, ey)[0, 1])
    res["sec1_partial_quality"] = {"comprehension_delta_r": partial(d.comprehension, d.delta_r_gpt2, d.reject_fraction),
                                   "comprehension_isc": partial(d.comprehension, d.isc, d.reject_fraction)}
    res["sec3_rho_isc_delta_r"] = float(stats.spearmanr(df.isc, df.delta_r_gpt2).correlation)
    du = d[d.authors_use]
    res["sec4_authors_use_subset"] = {"n": int(len(du)), "H1": pick(df[df.authors_use].delta_r_gpt2),
                                      "rho_comprehension_delta_r": float(stats.spearmanr(du.comprehension, du.delta_r_gpt2).correlation),
                                      "rho_comprehension_isc": float(stats.spearmanr(du.comprehension, du.isc).correlation)}
    res["means"] = {"isc": float(df.isc.mean()), "comprehension": float(d.comprehension.mean())}
    return res, df
