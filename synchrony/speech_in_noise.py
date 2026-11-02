"""Seeing the talker and rated intelligibility in the speech-in-noise data.

Passage-level synchrony by leave-one-passage-out spaces, the audiovisual statistic, and the within-listener association
of ratings with synchrony after double-centring.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from common.settings import dataset_root, rng as make_rng
from network_states.specification_search import signflip_maxt
from preprocessing import cnd_recordings as cnd, signal_operations as pp
from preprocessing.epoch_rejection import reject
from preprocessing.epochs import Trial
from statistical_tests.inference import summarise
from synchrony import intact_reversed_speech as b01
from synchrony.correlated_components import _cov_terms, corrca_from_terms, pair_cov, subject_isc, subject_shared_cov

ROOT = dataset_root("speech_in_noise")
FS = 128.0                          # the files' fs variable (512) is the acquisition rate
OFFSET = {"a": 0, "v": 15, "av": 30}  # file number = passage + offset
PASSAGES = range(1, 16)
SUBJECTS = range(1, 22)
BANDS = {"1-4": (1.0, 4.0), "0.5-4": (0.5, 4.0), "1-3": (1.0, 3.0)}
START_S, END_TRIM_S, MAX_BAD = 5.0, 2.0, 1 / 3
RATING_COND = {"av": 0, "a": 1}       # ratings: participants x passage x condition (1 = AV, 2 = A)


def load_trial(sub: int, cond: str, passage: int) -> np.ndarray:
    f = ROOT / "EEG" / f"sub{sub}" / f"sub{sub}_{cond}_{passage + OFFSET[cond]}.mat"
    return sio.loadmat(f, variable_names=["EEG"])["EEG"].T.astype(float)


def labels_128(paths) -> list[str]:
    f = next((paths.derived_root / "broderick").glob("forward_dataSub*.npz"))
    return [str(x) for x in np.load(f, allow_pickle=True)["labels"]]


def ratings(rows: str = "1-21") -> np.ndarray:
    """participants (EEG sub1..sub21) x passage x {av, a}. 22 rating rows vs 21 EEG participants;
    rows 1-21 are assumed to be sub1-sub21 (sensitivity: rows 2-22)."""
    s = sio.loadmat(ROOT / "Behavioural Data" / "Intelligibilty Ratings.mat")["score"].astype(float)
    return s[:21] if rows == "1-21" else s[1:22]


def prepare_one(sub: int, labels: list[str], cfg2: dict, seed: int, cache: Path) -> str:
    target = cache / f"sub{sub:02d}.npz"
    if target.exists():
        return target.name
    items = [(c, p, load_trial(sub, c, p)) for c in OFFSET for p in PASSAGES]
    info = cnd.montage_info(labels, FS, "biosemi128")
    hp = pp.fir_kernel(0.5, FS, cnd.fir_order(FS, 1.0), True)
    lp = pp.fir_kernel(40.0, FS, cnd.fir_order(FS, 5.0), False)
    ys = [pp.robust_detrend(x, 3, 3.0) for _, _, x in items]
    broad_raw = [pp.fir_filter(pp.fir_filter(y, hp), lp) for y in ys]
    a = int(START_S * FS)
    n_ep = [int((x.shape[1] / FS - START_S - END_TRIM_S) // 2) for _, _, x in items]
    wins = [(a, a + n * int(2 * FS)) for n in n_ep]
    bads = pp.bad_channels([bb[:, s:e] for bb, (s, e) in zip(broad_raw, wins)], labels, 5.0)
    op = cnd.spatial_operator(info, bads)
    rank = int((np.linalg.svd(op, compute_uv=False) > 1e-8).sum())
    step = int(round(FS / 64.0))
    fit = np.concatenate([(op @ bb[:, s:e])[:, ::step] for bb, (s, e) in zip(broad_raw, wins)], axis=1)
    info64 = cnd.montage_info(labels, FS / step, "biosemi128")
    proj, removed = cnd.ocular_ica(fit, info64, min(rank, 40), 4, 3.0, 3, int(make_rng(seed, "sin", sub).integers(2**31)))
    full = proj @ op
    trials = [Trial(subject=sub, condition=c, excerpt=p, label=f"{c}_{p}", presentation_index=j, engagement=np.nan,
                    familiarity=np.nan, sfreq=FS, t0=START_S, data=(full @ bb)[:, s:e])
              for j, ((c, p, _), bb, (s, e)) in enumerate(zip(items, broad_raw, wins))]
    res = reject(trials, labels, 2.0, seed=seed, max_bad_fraction=cfg2["max_bad_epoch_fraction"], method=cfg2["epoch_rejection"],
                 z_threshold=cfg2["faster_z"], channel_robust_z=cfg2["channel_robust_z"])
    arrays, meta = {}, []
    a64 = int(START_S * 64)
    for (c, p, _), y, t, n in zip(items, ys, trials, n_ep):
        ok = res.epoch_ok[t.key]
        excluded = bool((~ok).mean() > MAX_BAD)
        arrays[f"valid|{c}|{p}"] = np.repeat(ok & ~excluded, 128)
        for bname, (lo, hi) in BANDS.items():
            arrays[f"{bname}|{c}|{p}"] = cnd.band_64(full @ y, FS, lo, hi)[:, a64:a64 + n * 128].astype(np.float32)
        meta.append({"cond": c, "passage": p, "n_epochs": n, "reject_fraction": float((~ok).mean()), "excluded": excluded})
    np.savez(target, **arrays, meta=json.dumps({"bads": bads, "ica": [int(x) for x in removed], "trials": meta}))
    return target.name


def prepare_broderick_band(f: Path, bname: str, cfg2: dict, seed: int, cache: Path) -> str:
    """Broderick forward listener in an extra grid band."""
    target = cache / f"forward_{f.stem}_{bname}.npz"
    if target.exists():
        return target.name
    stim = cnd.load_stim(b01.cond_dir("forward") / "dataCND" / "dataStim.mat")
    durations = [len(e) / stim["fs"] for e in stim["envelope"]]
    prep = cnd.prepare_subject(cnd.load_subject(f), durations, "biosemi128", BANDS[bname], b01.START_S, b01.END_TRIM_S,
                               int(make_rng(seed, "brod", "forward", f.stem).integers(2**31)), cfg2, f.stem)
    np.savez(target, **{f"d{k}": d for k, d in enumerate(prep.runs)}, **{f"v{k}": v for k, v in enumerate(prep.valid)})
    return target.name


def load_all(cache: Path, band: str) -> dict:
    """{(cond, passage): {subject: (data, valid)}} for trials that passed exclusion."""
    out = {}
    for f in sorted(cache.glob("sub*.npz")):
        sub = int(f.stem[3:])
        z = np.load(f)
        for c in OFFSET:
            for p in PASSAGES:
                v = z[f"valid|{c}|{p}"]
                if v.any():
                    out.setdefault((c, p), {})[sub] = (z[f"{band}|{c}|{p}"].astype(float), v.astype(bool))
    return out


def stack(d: dict, subs: list[int]) -> tuple[np.ndarray, np.ndarray]:
    n = min(d[s][0].shape[1] for s in subs)
    return np.stack([d[s][0][:, :n] for s in subs]), np.stack([d[s][1][:n] for s in subs])


def speech_space(paths, band: str, ncomp: int) -> np.ndarray:
    cache = paths.derived_root / "broderick"
    if band == "1-4":
        _, runs = b01.load_prepared(cache, "forward")
    else:
        runs = {}
        for f in sorted((paths.derived_root / "broderick_bands").glob(f"forward_dataSub*_{band}.npz")):
            z = np.load(f)
            for k in range(sum(1 for n in z.files if n.startswith("d"))):
                v = z[f"v{k}"].astype(bool)
                if 1.0 - v.mean() > b01.MAX_BAD_FRACTION:
                    v = np.zeros_like(v)
                runs.setdefault(k, []).append((z[f"d{k}"].astype(float), v))
        runs = {k: (np.stack([d[:, :min(x.shape[1] for x, _ in it)] for d, _ in it]),
                    np.stack([v[:min(x.shape[1] for x, _ in it)] for _, v in it])) for k, it in runs.items()}
    Rw, Rb, N = 0.0, 0.0, 0
    for d, v in runs.values():
        w, bb = _cov_terms(d, v)
        Rw, Rb, N = Rw + w, Rb + bb, d.shape[0]
    return corrca_from_terms(Rw, Rb, N, 0.5, ncomp)[0]


def offdiag_mean(P: np.ndarray) -> float:
    ok = ~np.isnan(P[..., 0])
    return float(P[ok].sum(1).mean())


def passage_stats(data: dict, W: np.ndarray, p: int) -> tuple[dict, list[dict]] | tuple[None, None]:
    P = sorted(set(data.get(("a", p), {})) & set(data.get(("v", p), {})) & set(data.get(("av", p), {})))
    if len(P) < 4:
        return None, None
    X, ok, Y = {}, {}, {}
    for c in OFFSET:
        X[c], ok[c] = stack(data[(c, p)], P)
    n = min(X[c].shape[2] for c in OFFSET)
    for c in OFFSET:
        X[c], ok[c] = X[c][:, :, :n], ok[c][:, :n]
        Y[c] = np.matmul(W.T, X[c])
    Pm = {c: pair_cov(Y[c], ok[c]) for c in OFFSET}
    Pc = pair_cov(Y["a"], ok["a"], Y["v"], ok["v"])
    ya = [Y["a"][i][:, ok["a"][i]] for i in range(len(P))]
    T_A = float(np.mean([y.var(1).sum() for y in ya]))
    S = {c: offdiag_mean(Pm[c]) for c in OFFSET}
    C = offdiag_mean(Pc)
    isc = {c: subject_isc(X[c], W, ok[c]).sum(1) for c in OFFSET}
    row = {"passage": p, "n": len(P), "S_a": S["a"], "S_v": S["v"], "S_av": S["av"], "C_av": C, "T_a": T_A,
           "D": (S["av"] - (S["a"] + S["v"] + 2 * C)) / T_A,
           "isc_a": float(np.nanmean(isc["a"])), "isc_v": float(np.nanmean(isc["v"])), "isc_av": float(np.nanmean(isc["av"]))}
    ss = {c: subject_shared_cov(Pm[c]).sum(1) for c in OFFSET}
    sc = subject_shared_cov(Pc).sum(1)
    subj = [{"passage": p, "subject": s, "D_i": (ss["av"][j] - (ss["a"][j] + ss["v"][j] + 2 * sc[j])) / T_A}
            for j, s in enumerate(P)]
    return row, subj


def double_centre(M: np.ndarray, n_iter: int = 50) -> np.ndarray:
    """Remove row and column means from a matrix with missing values (iterated two-way additive fit)."""
    R = M.copy()
    for _ in range(n_iter):
        R = R - np.nanmean(R, axis=1, keepdims=True)
        R = R - np.nanmean(R, axis=0, keepdims=True)
    return R


def graded(data: dict, W: np.ndarray, cond: str, rows: str) -> tuple[dict, pd.DataFrame]:
    I = np.full((21, 15), np.nan)
    for p in PASSAGES:
        d = data.get((cond, p), {})
        subs = sorted(d)
        if len(subs) < 4:
            continue
        x, ok = stack(d, subs)
        v = subject_isc(x, W, ok).sum(1)
        for s, val in zip(subs, v):
            I[s - 1, p - 1] = val
    R = ratings(rows)[:, :, RATING_COND[cond]]
    R = np.where(np.isnan(I), np.nan, R)
    Ic, Rc = double_centre(I), double_centre(R)
    z = []
    for i in range(21):
        m = ~np.isnan(Ic[i]) & ~np.isnan(Rc[i])
        if m.sum() >= 5 and Rc[i][m].std() > 0:
            z.append(np.arctanh(np.clip(np.corrcoef(Ic[i][m], Rc[i][m])[0, 1], -0.999, 0.999)))
    long = pd.DataFrame([{"subject": i + 1, "passage": p + 1, "isc": I[i, p], "rating": R[i, p]}
                         for i in range(21) for p in range(15) if not np.isnan(I[i, p])])
    return summarise(np.array(z), "greater"), long


def pick(s: dict) -> dict:
    return {k: s[k] for k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}


def seeing_the_talker(paths, cache: Path, out: Path) -> dict:
    res = {}
    data = load_all(cache, "1-4")
    W = speech_space(paths, "1-4", 3)
    rows, subj = [], []
    for p in PASSAGES:
        r, s = passage_stats(data, W, p)
        if r is not None:
            rows.append(r)
            subj += s
    df = pd.DataFrame(rows)
    df.to_csv(out / "speech_in_noise_passages.csv", index=False)
    pd.DataFrame(subj).to_csv(out / "speech_in_noise_participants.csv", index=False)
    res["primary_items_D"] = pick(summarise(df.D.to_numpy(), "greater"))
    res["passage_means"] = df.drop(columns="passage").mean().round(5).to_dict()
    res["sec1_isc_gain_av_minus_a"] = pick(summarise((df.isc_av - df.isc_a).to_numpy(), "greater"))
    res["sec2_isc_visual_only"] = pick(summarise(df.isc_v.to_numpy(), "two-sided"))
    for rows_map in ("1-21", "2-22"):
        for cond in ("a", "av"):
            s, long = graded(data, W, cond, rows_map)
            res[f"sec3_graded_{cond}_rows{rows_map}"] = pick(s)
            if rows_map == "1-21":
                long.to_csv(out / f"speech_in_noise_graded_{cond}.csv", index=False)
    sc = ratings("1-21")
    all22 = sio.loadmat(ROOT / "Behavioural Data" / "Intelligibilty Ratings.mat")["score"].astype(float)
    res["sec4_rating_av_minus_a_passages"] = pick(summarise((all22[:, :, 0] - all22[:, :, 1]).mean(0), "greater"))
    res["sec4_rating_means_pct_bin"] = {"av": float(sc[:, :, 0].mean()), "a": float(sc[:, :, 1].mean())}
    part = pd.DataFrame(subj).groupby("subject").D_i.mean()
    res["sec5_participants_D"] = pick(summarise(part.to_numpy(), "greater"))
    cols, grid = [], []
    for bname in BANDS:
        d = data if bname == "1-4" else load_all(cache, bname)
        for nc in (1, 3, 5):
            if bname == "1-4" and nc == 3:
                v = df.set_index("passage").D
            else:
                Wg = speech_space(paths, bname, nc)
                v = pd.Series({p: r["D"] for p in PASSAGES if (r := passage_stats(d, Wg, p)[0]) is not None})
            cols.append(v.rename(f"{bname}|{nc}"))
            grid.append({"band": bname, "ncomp": nc})
    M = pd.concat(cols, axis=1).dropna()
    M.to_csv(out / "speech_in_noise_grid.csv")
    _, p_unc, p_fwe = signflip_maxt(M.to_numpy(), alternative="greater")
    res["sec6_grid"] = [g | {"dz": float(M.iloc[:, j].mean() / M.iloc[:, j].std(ddof=1)), "p_unc": float(p_unc[j]),
                             "p_fwe": float(p_fwe[j])} for j, g in enumerate(grid)]
    within = []
    for p in PASSAGES:
        Rw, Rb = 0.0, 0.0
        for q in PASSAGES:
            dq = data.get(("a", q), {})
            if q == p or len(dq) < 3:
                continue
            w, bb = _cov_terms(*stack(dq, sorted(dq)))
            Rw, Rb = Rw + w, Rb + bb
        r, _ = passage_stats(data, corrca_from_terms(Rw, Rb, 21, 0.5, 3)[0], p)
        if r is not None:
            within.append(r["D"])
    res["sec7_within_dataset_space_D"] = pick(summarise(np.array(within), "greater"))
    return res
