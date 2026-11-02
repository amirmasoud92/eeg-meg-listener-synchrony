"""Intact against time-reversed speech (Broderick et al., 2018) and the speech-defined space.

Run-level statistic with leave-one-run-out filters, the intact group subsampled to the size of the reversed group, and
the speech-defined fixed space applied to the Aalborg data.
"""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common.settings import dataset_root, rng as make_rng
from network_states.source_model import make_info
from preprocessing import cnd_recordings as cnd
from readers import etard_release as etard
from readers.aalborg_release import load_channel_names
from statistical_tests.inference import summarise
from synchrony import understood_foreign
from synchrony.correlated_components import _cov_terms, corrca_from_terms, subject_isc

ROOT = dataset_root("broderick")
CONDS = {"forward": "LalorNatSpeech", "reversed": "LalorNatSpeechReverse"}
BAND, START_S, END_TRIM_S, N_SUBSETS = (1.0, 4.0), 5.0, 2.0, 200
MAX_BAD_FRACTION = 1 / 3   # a run is excluded when more than a third of its epochs are rejected


def cond_dir(cond: str) -> Path:
    base = ROOT / CONDS[cond]
    return base if (base / "dataCND").exists() else next(ROOT.glob(f"**/{CONDS[cond]}"))


def subject_files(cond: str) -> list[Path]:
    return sorted((cond_dir(cond) / "dataCND").glob("dataSub*.mat"), key=lambda p: int(p.stem.replace("dataSub", "")))


def prepare_one(cond: str, f: Path, cfg2: dict, seed: int, cache: Path) -> str:
    target = cache / f"{cond}_{f.stem}.npz"
    if target.exists():
        return target.name
    stim = cnd.load_stim(cond_dir(cond) / "dataCND" / "dataStim.mat")
    durations = [len(e) / stim["fs"] for e in stim["envelope"]]
    sub = cnd.load_subject(f)
    prep = cnd.prepare_subject(sub, durations, "biosemi128", BAND, START_S, END_TRIM_S,
                               int(make_rng(seed, "brod", cond, f.stem).integers(2**31)), cfg2, f.stem)
    np.savez(target, **{f"d{k}": d for k, d in enumerate(prep.runs)}, **{f"v{k}": v for k, v in enumerate(prep.valid)},
             labels=np.array(sub["labels"]), bads=np.array(prep.bads, dtype=object), ica=np.array(prep.ica_removed),
             reject_fraction=prep.reject_fraction, allow_pickle=True)
    return target.name


def load_prepared(cache: Path, cond: str) -> tuple[list[str], dict]:
    files = sorted(cache.glob(f"{cond}_dataSub*.npz"), key=lambda p: int(p.stem.split("dataSub")[1]))
    subs, runs = [], {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        subs.append(f.stem)
        for k in range(sum(1 for n in z.files if n.startswith("d"))):
            v = z[f"v{k}"].astype(bool)
            if 1.0 - v.mean() > MAX_BAD_FRACTION:
                v = np.zeros_like(v)
            runs.setdefault(k, []).append((z[f"d{k}"].astype(float), v))
    arr = {}
    for k, items in runs.items():
        n = min(d.shape[1] for d, _ in items)
        arr[k] = (np.stack([d[:, :n] for d, _ in items]), np.stack([v[:n] for _, v in items]))
    return subs, arr


def pair_terms(data: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-subject Gram matrices G (n x C x C) and pairwise cross products (n x n x C x C), centred and
    masked exactly as ``correlated_components._cov_terms``."""
    X = data - np.nanmean(data, axis=2, keepdims=True)
    X = np.where(valid[:, None, :], np.nan_to_num(X), 0.0)
    n, C, T = X.shape
    Xf = X.reshape(n * C, T)
    cross = (Xf @ Xf.T).reshape(n, C, n, C).transpose(0, 2, 1, 3)
    return np.einsum("iicd->icd", cross), cross


def subset_terms(G, cross, idx) -> tuple[np.ndarray, np.ndarray]:
    Rw = G[idx].sum(0)
    Rt = cross[np.ix_(idx, idx)].sum((0, 1))
    return Rw, Rt - Rw


def _run_one(fit_name, parts, r, terms_F, terms_R, Fr, Rr, subsets, idxR, size, nF) -> dict:
    GF, XF = terms_F
    wR, bR = terms_R
    accF, accR = [], []
    sumF, cntF, sumR = np.zeros(nF), np.zeros(nF), np.zeros(len(idxR))
    for S in subsets:
        Rw, Rb = 0.0, 0.0
        if "F" in parts:
            w, b = subset_terms(GF, XF, S)
            Rw, Rb = Rw + w, Rb + b
        if "R" in parts:
            Rw, Rb = Rw + wR, Rb + bR
        W, _ = corrca_from_terms(Rw, Rb, size, 0.5, 3)
        iF = subject_isc(Fr[0][S], W, Fr[1][S]).sum(1)
        iR = subject_isc(Rr[0][idxR], W, Rr[1][idxR]).sum(1)
        accF.append(np.nanmean(iF))
        accR.append(np.nanmean(iR))
        sumF[S] += np.nan_to_num(iF)
        cntF[S] += np.isfinite(iF)
        sumR += np.nan_to_num(iR)
    return {"fit": fit_name, "run": r, "isc_forward": float(np.mean(accF)), "isc_reversed": float(np.mean(accR)),
            "subject_forward": (sumF / np.maximum(cntF, 1)).tolist(), "subject_reversed": (sumR / len(subsets)).tolist()}


def intact_vs_reversed(cache: Path, out: Path, seed: int, residual: bool = False, n_jobs: int = 10) -> dict:
    subsF, F = load_prepared(cache, "forward")
    subsR, R = load_prepared(cache, "reversed")
    if residual:
        F, R = remove_envelope(cache, "forward", F), remove_envelope(cache, "reversed", R)
    runs = sorted(set(F) & set(R))
    nF, nR = len(subsF), len(subsR)
    size = min(nF, nR)
    tF = {r: pair_terms(*F[r]) for r in runs}
    tR = {r: pair_terms(*R[r]) for r in runs}
    g = np.random.default_rng(seed)
    subsets = [np.sort(g.choice(nF, size, replace=False)) for _ in range(N_SUBSETS if nF > size else 1)]
    idxR = np.arange(nR)[: size] if nR > size else np.arange(nR)
    fits = {"pooled": ("F", "R"), "reversed_only": ("R",), "forward_only": ("F",)}
    totF = (sum(tF[q][0] for q in runs), sum(tF[q][1] for q in runs))
    totR = (sum(tR[q][0] for q in runs), sum(tR[q][1] for q in runs))
    jobs = [(fit_name, parts, r, (totF[0] - tF[r][0], totF[1] - tF[r][1]),
             subset_terms(totR[0] - tR[r][0], totR[1] - tR[r][1], idxR), F[r], R[r])
            for fit_name, parts in fits.items() for r in runs]
    rows = Parallel(n_jobs=n_jobs)(delayed(_run_one)(*j, subsets, idxR, size, nF) for j in jobs)
    df = pd.DataFrame(rows)
    res = {"n_forward": nF, "n_reversed": nR, "group_size": size, "n_subsets": len(subsets), "n_runs": len(runs)}
    for fit_name, g2 in df.groupby("fit"):
        diff = (g2.isc_forward - g2.isc_reversed).to_numpy()
        s = summarise(diff, "greater")
        sf = np.nanmean(np.vstack(g2.subject_forward.to_list()), axis=0)
        sr = np.nanmean(np.vstack(g2.subject_reversed.to_list()), axis=0)
        obs = sf.mean() - sr.mean()
        pool = np.r_[sf, sr]
        perm = np.random.default_rng(seed)
        null = np.array([(lambda p: p[:sf.size].mean() - p[sf.size:].mean())(perm.permutation(pool)) for _ in range(20000)])
        res[fit_name] = {"runs_forward_minus_reversed": {k: s[k] for k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")},
                         "mean_isc_forward": float(g2.isc_forward.mean()), "mean_isc_reversed": float(g2.isc_reversed.mean()),
                         "participant_level": {"forward_mean": float(sf.mean()), "reversed_mean": float(sr.mean()),
                                               "diff": float(obs), "p_one_sided": float((1 + (null >= obs).sum()) / (1 + null.size))}}
    df.drop(columns=["subject_forward", "subject_reversed"]).to_csv(out / f"intact_reversed_runs{'_residual' if residual else ''}.csv", index=False)
    return res


def remove_envelope(cache: Path, cond: str, arr: dict) -> dict:
    """Subtract the cross-validated (leave-one-run-out) linear envelope prediction, lags 0-500 ms."""
    stim = cnd.load_stim(cond_dir(cond) / "dataCND" / "dataStim.mat")
    envs = {}
    for r, e in enumerate(stim["envelope"]):
        eb = cnd.band_64(e[None], stim["fs"], *BAND)[0]
        a = int(START_S * 64)
        envs[r] = eb[a: a + arr[r][0].shape[2]]
    n_lags = 33
    lag = lambda e: np.column_stack([np.r_[np.zeros(k), e[: e.size - k]] for k in range(n_lags)])
    out = {}
    runs = sorted(arr)
    n_sub = arr[runs[0]][0].shape[0]
    resid = {r: arr[r][0].copy() for r in runs}
    for i in range(n_sub):
        for r in runs:
            A = np.vstack([lag(envs[q])[arr[q][1][i]] for q in runs if q != r])
            B = np.vstack([arr[q][0][i][:, arr[q][1][i]].T for q in runs if q != r])
            B = B - B.mean(0)
            G = A.T @ A
            W = np.linalg.solve(G + np.trace(G) / n_lags * np.eye(n_lags), A.T @ B)
            resid[r][i] = arr[r][0][i] - (lag(envs[r]) @ W).T
    for r in runs:
        out[r] = (resid[r], arr[r][1])
    return out


def aalborg_isc(W: np.ndarray, arr: dict, labels: dict, subj: list, channel_idx=None) -> pd.DataFrame:
    rows = []
    for k, (d, v) in arr.items():
        if labels[k] not in ("MC", "MD", "MF", "SD", "SF"):
            continue
        dd = d.astype(float) if channel_idx is None else d[:, channel_idx].astype(float)
        isc = subject_isc(dd, W, v)
        rows += [{"subject": s, "stimulus": k, "condition": labels[k], "isc": float(np.nansum(isc[i]))}
                 for i, s in enumerate(subj) if v[i].any()]
    return pd.DataFrame(rows)


def contrasts(df: pd.DataFrame) -> dict:
    w = df.groupby(["subject", "condition"]).isc.mean().unstack()
    res = {}
    for name, a, b, alt in (("MD-MF", "MD", "MF", "greater"), ("SD-SF", "SD", "SF", "greater"),
                            ("MD-MC", "MD", "MC", "two-sided"), ("MF-MC", "MF", "MC", "two-sided")):
        s = summarise((w[a] - w[b]).to_numpy(), alt)
        res[name] = {k: s[k] for k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")} | {"alternative": alt}
    it = df.groupby("stimulus").isc.mean()
    songs = sorted({k.split("_")[0] for k in it.index if k.endswith("MD.")})
    diffs = np.array([it[f"{s}_MD."] - it[f"{s}_MF."] for s in songs])
    import itertools
    null = np.array([np.mean(diffs * np.array(sg)) for sg in itertools.product([1, -1], repeat=diffs.size)])
    res["item_paired_MD-MF"] = {"diffs": diffs.round(4).tolist(), "p_one_sided": float(np.mean(null >= diffs.mean() - 1e-12))}
    res["condition_means"] = w.mean().round(4).to_dict()
    return res


def understood_vs_foreign_in_speech_space(cache: Path, out: Path, paths, cfg) -> dict:
    subsF, F = load_prepared(cache, "forward")
    labels_b = [str(x) for x in np.load(next(cache.glob("forward_dataSub*.npz")), allow_pickle=True)["labels"]]
    Rw, Rb = 0.0, 0.0
    for r, (d, v) in F.items():
        w, b = _cov_terms(d, v)
        Rw, Rb = Rw + w, Rb + b
    aal_names = load_channel_names(paths.channel_list)
    info_b = cnd.montage_info(labels_b, 64.0, "biosemi128")
    target = make_info(aal_names, 64.0).get_montage()
    probe = mne.io.RawArray(np.eye(len(labels_b)), info_b, verbose="error")
    interp = probe.interpolate_to(target, method="spline", reg=1e-5)
    if interp.ch_names != list(aal_names):
        order = [interp.ch_names.index(c) for c in aal_names]
        M = interp.get_data()[order]
    else:
        M = interp.get_data()
    H = np.eye(M.shape[0]) - 1.0 / M.shape[0]
    M = H @ M
    W, _ = corrca_from_terms(M @ Rw @ M.T, M @ Rb @ M.T, len(subsF), 0.5, 3)
    labels = understood_foreign.song_labels(paths)
    arr, subj = understood_foreign.arrays(paths, cfg, "aalborg", BAND)
    df = aalborg_isc(W, arr, labels, subj)
    df.to_csv(out / "aalborg_isc_speech_space.csv", index=False)
    res = {"primary_broderick_space": contrasts(df), "interpolation_matrix_shape": list(M.shape)}
    cfg_all = etard.load_config()
    earr, esubj = understood_foreign.arrays(paths, cfg, "etard", BAND, cfg_all.section("network"))
    e_names = nw_channels(paths)
    common = [c for c in aal_names if c in e_names]
    ie, ia = [e_names.index(c) for c in common], [aal_names.index(c) for c in common]
    Rw, Rb = 0.0, 0.0
    for k, (d, v) in earr.items():
        if k.split("_")[0] in ("clean", "lb", "mb", "hb"):
            w, b = _cov_terms(d[:, ie].astype(float), v)
            Rw, Rb = Rw + w, Rb + b
    W2, _ = corrca_from_terms(Rw, Rb, len(esubj), 0.5, 3)
    df2 = aalborg_isc(W2, arr, labels, subj, channel_idx=ia)
    df2.to_csv(out / "testB_aalborg_isc_etard_space.csv", index=False)
    res["secondary_etard_space"] = contrasts(df2) | {"n_common_channels": len(common)}
    return res


def nw_channels(paths) -> list[str]:
    from preprocessing import etard_recordings as nw
    cfg_n = etard.load_config().section("network")
    return nw.EtardDataset(paths.derived_root / "etard" / "preprocessed", cfg_n["participants"]).channels
