"""Attention in the cocktail-party data: same-story against other-story synchrony.

Each listener's component time course is correlated with the group attending the same story and with the group
attending the other story (groups matched by subsampling, 100 subsets), and run-wise comprehension is related to
synchrony within listeners.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from common.settings import dataset_root, rng as make_rng
from preprocessing import cnd_recordings as cnd, signal_operations as pp
from preprocessing.epoch_rejection import reject
from preprocessing.epochs import Trial
from statistical_tests.inference import summarise
from synchrony import speech_in_noise as d01

ROOT = dataset_root("cocktail_party")
FS = 128.0
SUBJECTS = range(1, 34)
RUNS = range(1, 31)
GROUP = {s: ("20000" if s <= 17 else "Journey") for s in SUBJECTS}   # readme: 1-17 attend left (20000), 18-33 right
START_S, END_TRIM_S, MAX_BAD, N_SUBSETS = 5.0, 2.0, 1 / 3, 100


def trial_path(sub: int, run: int) -> Path:
    return ROOT / "EEG" / f"Subject{sub}" / f"Subject{sub}_Run{run}.mat"


def load_trial(sub: int, run: int) -> np.ndarray:
    x = sio.loadmat(trial_path(sub, run), variable_names=["eegData"])["eegData"].astype(float)
    return x.T if x.shape[1] == 128 else x


def scores() -> np.ndarray:
    """listeners x runs x (attended, unattended), fraction correct."""
    return sio.loadmat(ROOT / "Behavioural Data" / "Comprehension Scores.mat")["score"].astype(float)


def prepare_one(sub: int, labels: list[str], cfg2: dict, seed: int, cache: Path) -> str:
    target = cache / f"sub{sub:02d}.npz"
    if target.exists():
        return target.name
    items = [(r, load_trial(sub, r)) for r in RUNS if trial_path(sub, r).exists()]
    info = cnd.montage_info(labels, FS, "biosemi128")
    hp = pp.fir_kernel(0.5, FS, cnd.fir_order(FS, 1.0), True)
    lp = pp.fir_kernel(40.0, FS, cnd.fir_order(FS, 5.0), False)
    ys = [pp.robust_detrend(x, 3, 3.0) for _, x in items]
    broad_raw = [pp.fir_filter(pp.fir_filter(y, hp), lp) for y in ys]
    a = int(START_S * FS)
    n_ep = [int((x.shape[1] / FS - START_S - END_TRIM_S) // 2) for _, x in items]
    wins = [(a, a + n * int(2 * FS)) for n in n_ep]
    bads = pp.bad_channels([bb[:, s:e] for bb, (s, e) in zip(broad_raw, wins)], labels, 5.0)
    op = cnd.spatial_operator(info, bads)
    rank = int((np.linalg.svd(op, compute_uv=False) > 1e-8).sum())
    fit = np.concatenate([(op @ bb[:, s:e])[:, ::2] for bb, (s, e) in zip(broad_raw, wins)], axis=1)
    proj, removed = cnd.ocular_ica(fit, cnd.montage_info(labels, 64.0, "biosemi128"), min(rank, 40), 4, 3.0, 3,
                                   int(make_rng(seed, "cocktail", sub).integers(2**31)))
    full = proj @ op
    trials = [Trial(subject=sub, condition="run", excerpt=r, label=f"run{r}", presentation_index=j, engagement=np.nan,
                    familiarity=np.nan, sfreq=FS, t0=START_S, data=(full @ bb)[:, s:e])
              for j, ((r, _), bb, (s, e)) in enumerate(zip(items, broad_raw, wins))]
    res = reject(trials, labels, 2.0, seed=seed, max_bad_fraction=cfg2["max_bad_epoch_fraction"], method=cfg2["epoch_rejection"],
                 z_threshold=cfg2["faster_z"], channel_robust_z=cfg2["channel_robust_z"])
    arrays, meta = {}, []
    a64 = int(START_S * 64)
    for (r, _), y, t, n in zip(items, ys, trials, n_ep):
        ok = res.epoch_ok[t.key]
        excluded = bool((~ok).mean() > MAX_BAD)
        arrays[f"valid|{r}"] = np.repeat(ok & ~excluded, 128)
        arrays[f"1-4|{r}"] = cnd.band_64(full @ y, FS, 1.0, 4.0)[:, a64:a64 + n * 128].astype(np.float32)
        meta.append({"run": r, "reject_fraction": float((~ok).mean()), "excluded": excluded})
    np.savez(target, **arrays, meta=json.dumps({"bads": bads, "ica": [int(x) for x in removed], "trials": meta}))
    return target.name


def load_all(cache: Path) -> dict:
    """{run: {listener: (data, valid)}} for trials that passed exclusion."""
    out = {}
    for f in sorted(cache.glob("sub*.npz")):
        sub = int(f.stem[3:])
        z = np.load(f)
        for r in RUNS:
            if f"valid|{r}" in z.files and z[f"valid|{r}"].any():
                out.setdefault(r, {})[sub] = (z[f"1-4|{r}"].astype(float), z[f"valid|{r}"].astype(bool))
    return out


def isc_to_group(y: np.ndarray, ok: np.ndarray, Yg: np.ndarray, okg: np.ndarray) -> float:
    """Correlation (summed over components) of one listener's component time courses with the mean of a group's,
    over samples valid for the listener and for at least half of the group."""
    w = okg.astype(float)
    cnt = w.sum(0)
    mean = np.einsum("nkt,nt->kt", Yg, w) / np.maximum(cnt, 1)
    sel = ok & (cnt >= max(1, Yg.shape[0] / 2))
    if sel.sum() < 128:
        return np.nan
    a, b = y[:, sel], mean[:, sel]
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return float(((a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))).sum())


def run_isc(d: dict, W: np.ndarray, g: np.random.Generator) -> dict:
    subs = sorted(d)
    n = min(d[s][0].shape[1] for s in subs)
    Y = {s: W.T @ d[s][0][:, :n] for s in subs}
    ok = {s: d[s][1][:n] for s in subs}
    out = {}
    for i in subs:
        same = [s for s in subs if s != i and GROUP[s] == GROUP[i]]
        other = [s for s in subs if GROUP[s] != GROUP[i]]
        k = min(len(same), len(other))
        if k < 3:
            continue
        vals = {"same": [], "other": []}
        for _ in range(N_SUBSETS if max(len(same), len(other)) > k else 1):
            for name, grp in (("same", same), ("other", other)):
                pick = sorted(g.choice(grp, k, replace=False)) if len(grp) > k else grp
                vals[name].append(isc_to_group(Y[i], ok[i], np.stack([Y[s] for s in pick]), np.stack([ok[s] for s in pick])))
        out[i] = {"same": float(np.nanmean(vals["same"])), "other": float(np.nanmean(vals["other"])), "k": k}
    return out


def graded(I: np.ndarray, S: np.ndarray, min_distinct: int = 3) -> tuple[dict, int]:
    S = np.where(np.isnan(I), np.nan, S)
    Ic, Sc = d01.double_centre(I), d01.double_centre(S)
    z = []
    for i in range(I.shape[0]):
        m = ~np.isnan(Ic[i]) & ~np.isnan(Sc[i])
        if m.sum() >= 5 and len(np.unique(S[i][m])) >= min_distinct and Sc[i][m].std() > 0:
            z.append(np.arctanh(np.clip(np.corrcoef(Ic[i][m], Sc[i][m])[0, 1], -0.999, 0.999)))
    return d01.pick(summarise(np.array(z), "greater")), len(z)


def attention_test(paths, cache: Path, out: Path, seed: int) -> dict:
    data = load_all(cache)
    W = d01.speech_space(paths, "1-4", 3)
    g = np.random.default_rng([seed, 5])
    rows = []
    for r in RUNS:
        if r in data and len(data[r]) >= 8:
            for i, v in run_isc(data[r], W, g).items():
                rows.append({"run": r, "listener": i, "group": GROUP[i], **v})
    df = pd.DataFrame(rows)
    df.to_csv(out / "cocktail_party_isc.csv", index=False)
    res = {}
    per_run = df.groupby("run").apply(lambda t: (t.same - t.other).mean(), include_groups=False)
    res["runs_same_minus_other"] = d01.pick(summarise(per_run.to_numpy(), "greater"))
    res["mean_isc_same_other"] = [float(df.same.mean()), float(df["other"].mean())]
    sc = scores()
    I_same = np.full((33, 30), np.nan)
    I_other = np.full((33, 30), np.nan)
    for _, rw in df.iterrows():
        I_same[int(rw.listener) - 1, int(rw.run) - 1] = rw.same
        I_other[int(rw.listener) - 1, int(rw.run) - 1] = rw["other"]
    res["graded_attended_score_vs_isc_same"], res["n_listeners"] = graded(I_same, sc[:, :, 0])
    res["graded_attended_score_vs_isc_other"], _ = graded(I_other, sc[:, :, 0])
    res["comprehension_means"] = {"attended": float(np.nanmean(sc[:, :, 0])), "unattended": float(np.nanmean(sc[:, :, 1]))}
    return res
