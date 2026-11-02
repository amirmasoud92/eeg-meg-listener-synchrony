"""Attention test after removal of the responses to both story envelopes.

Per listener, a leave-one-run-out ridge model predicts the 1-4 Hz EEG from both streams' envelopes and onsets, and the
prediction is subtracted. Correlated components are fitted on the residuals, leave-one-run-out, and the same-minus-other
statistic is recomputed on each held-out run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from joblib import Parallel, delayed

from common.settings import load_analysis, load_paths
from preprocessing import cnd_recordings as cnd
from statistical_tests.inference import summarise
from synchrony import acoustic_removal as f, cocktail_party as e, intact_reversed_speech as b, speech_in_noise as d01
from synchrony.correlated_components import corrca_from_terms

ENV = e.ROOT / "Stimuli" / "Envelopes"
STREAMS = ("20000", "Journey")


def env_features(run: int) -> np.ndarray:
    """4 x samples at 64 Hz (1-4 Hz): envelope and onset of each stream, from the released 128 Hz envelopes."""
    rows = []
    for s in STREAMS:
        m = sio.loadmat(ENV / s / f"{s}_{run}_env.mat", squeeze_me=True)
        x = np.asarray(m["envelope"], float)
        rows += [x, np.maximum(np.diff(x, prepend=x[:1]), 0)]
    n = min(r.size for r in rows)
    return cnd.band_64(np.vstack([r[:n] for r in rows]), float(m["fsEnv"]), 1.0, 4.0)


def residuals_listener(sub: int, cache: Path, feats: dict, out_dir: Path) -> dict:
    """Leave-one-run-out envelope model for one listener; residual EEG per run (saved) and the chosen ridge."""
    target = out_dir / f"sub{sub:02d}.npz"
    if target.exists():
        return json.loads(str(np.load(target)["meta"]))
    z = np.load(cache / f"sub{sub:02d}.npz")
    runs = [r for r in e.RUNS if f"valid|{r}" in z.files and z[f"valid|{r}"].any()]
    a64 = int(e.START_S * 64)
    Xs, Ys, Vs, Gs, Cs = {}, {}, {}, {}, {}
    for r in runs:
        Y = z[f"1-4|{r}"].astype(float)
        v = z[f"valid|{r}"].astype(bool)
        T = Y.shape[1]
        F = feats[r][:, a64:a64 + T]
        if F.shape[1] < T:
            F = np.pad(F, ((0, 0), (0, T - F.shape[1])))
        F = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-12)
        X = f.lagged(F, T, f.LAG_STEP).astype(np.float64)
        Xs[r], Ys[r], Vs[r] = X, Y, v
        Gs[r], Cs[r] = X[v].T @ X[v], X[v].T @ Y[:, v].T
    G, C = sum(Gs.values()), sum(Cs.values())
    preds = {lam: {} for lam in f.RIDGE_GRID}
    for r in runs:
        ev, V = np.linalg.eigh(G - Gs[r])
        VC, Xr, m = V.T @ (C - Cs[r]), Xs[r] @ V, ev.mean()
        for lam in f.RIDGE_GRID:
            preds[lam][r] = np.zeros_like(Ys[r]) if np.isinf(lam) else (Xr @ (VC / (ev + lam * m)[:, None])).T
    sse = {lam: sum(float(((Ys[r][:, Vs[r]] - preds[lam][r][:, Vs[r]]) ** 2).sum()) for r in runs) for lam in f.RIDGE_GRID}
    best = min(sse, key=sse.get)
    tot = sum(float((Ys[r][:, Vs[r]] ** 2).sum()) for r in runs)
    meta = {"sub": sub, "lambda_x_mean_eig": float(best), "frac_removed": 1 - sse[best] / tot, "runs": runs}
    np.savez(target, **{f"r{r}": (Ys[r] - preds[best][r]).astype(np.float32) for r in runs},
             **{f"v{r}": Vs[r] for r in runs}, meta=json.dumps(meta))
    return meta


def load_level(cache: Path, resid_dir: Path | None) -> dict:
    """{run: {listener: (data, valid)}} from the preprocessed runs (resid_dir None) or from the residuals."""
    if resid_dir is None:
        return e.load_all(cache)
    out = {}
    for fz in sorted(resid_dir.glob("sub*.npz")):
        sub = int(fz.stem[3:])
        z = np.load(fz)
        for r in json.loads(str(z["meta"]))["runs"]:
            out.setdefault(r, {})[sub] = (z[f"r{r}"].astype(float), z[f"v{r}"].astype(bool))
    return out


def run_terms(d: dict) -> tuple[np.ndarray, np.ndarray, int]:
    subs = sorted(d)
    n = min(d[s][0].shape[1] for s in subs)
    X = np.stack([d[s][0][:, :n] for s in subs])
    V = np.stack([d[s][1][:n] for s in subs])
    G, cross = b.pair_terms(X, V)
    Rw = G.sum(0)
    return Rw, cross.sum((0, 1)) - Rw, len(subs)


def loro_isc(data: dict, seed: int, W_fixed: np.ndarray | None = None) -> pd.DataFrame:
    """Same/other ISC per run and listener; filters refitted leave-one-run-out (or fixed)."""
    runs = [r for r in sorted(data) if len(data[r]) >= 8]
    terms = {r: run_terms(data[r]) for r in runs} if W_fixed is None else {}
    g = np.random.default_rng([seed, 6, 3])
    rows = []
    for r in runs:
        if W_fixed is None:
            Rw = sum(terms[q][0] for q in runs if q != r)
            Rb = sum(terms[q][1] for q in runs if q != r)
            N = np.mean([terms[q][2] for q in runs if q != r])
            W, _ = corrca_from_terms(Rw, Rb, N, 0.5, 3)
        else:
            W = W_fixed
        for i, v in e.run_isc(data[r], W, g).items():
            rows.append({"run": r, "listener": i, "group": e.GROUP[i], **v})
    return pd.DataFrame(rows)


def summarise_level(df: pd.DataFrame) -> dict:
    pick = lambda x, alt: {k: v for k, v in summarise(np.asarray(x, float), alt).items() if k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}
    per_run = df.groupby("run").apply(lambda t: (t.same - t.other).mean(), include_groups=False)
    level = df.groupby("run").apply(lambda t: ((t.same + t["other"]) / 2).mean(), include_groups=False)
    return {"isc_level_runs": pick(level, "greater"), "same_minus_other_runs": pick(per_run, "greater"),
            "mean_same_other": [float(df.same.mean()), float(df["other"].mean())]}


def attention_after_envelope_removal(n_jobs: int = 6):
    paths, cfg = load_paths(), load_analysis()
    cache = paths.derived_root / "cocktail_party"
    rdir = paths.derived_root / "cocktail_party_residuals"
    out = paths.results_root
    feats = {r: env_features(r) for r in e.RUNS}
    subs = sorted(int(p.stem[3:]) for p in cache.glob("sub*.npz"))
    metas = Parallel(n_jobs=n_jobs)(delayed(residuals_listener)(s, cache, feats, rdir) for s in subs)
    res = {"variance_removed": {"mean": float(np.mean([m["frac_removed"] for m in metas])),
                                "range": [float(min(m["frac_removed"] for m in metas)), float(max(m["frac_removed"] for m in metas))]},
           "ridges": {m["sub"]: m["lambda_x_mean_eig"] for m in metas}}
    resid = load_level(cache, rdir)
    df_res = loro_isc(resid, cfg.project.seed)
    df_res.to_csv(out / "word_level_residual_isc.csv", index=False)
    res["residual_refit"] = summarise_level(df_res)
    W_speech = d01.speech_space(paths, "1-4", 3)
    df_sp = loro_isc(resid, cfg.project.seed, W_fixed=W_speech)
    res["residual_speech_space"] = summarise_level(df_sp)
    orig = load_level(cache, None)
    df_orig = loro_isc(orig, cfg.project.seed)
    df_orig.to_csv(out / "word_level_refit_original_isc.csv", index=False)
    res["refit_original"] = summarise_level(df_orig)
    ok = res["residual_refit"]["isc_level_runs"]["p_signflip"] < 0.05
    res["interpretable"] = bool(ok)
    (out / "word_level.json").write_text(json.dumps(res, indent=2, default=float))
