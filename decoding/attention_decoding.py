"""Attention in the cocktail-party data, from envelope tracking and decoding.

Forward model: per listener, leave-one-run-out ridge from the attended and from the ignored envelope (lags 0-500 ms) to
the 1-4 Hz EEG. Backward model (O'Sullivan et al., 2015): all channels, lags 0-250 ms, reconstructing the attended
envelope; a run is decoded correctly when the reconstruction correlates more with the attended than the ignored envelope.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.io as sio
from joblib import Parallel, delayed

from common.settings import load_analysis, load_paths
from preprocessing import cnd_recordings as cnd
from statistical_tests.inference import summarise
from synchrony import cocktail_party as e
from synchrony.correlated_components import _cov_terms, corrca_from_terms

N_LAGS = 33


def envelope(story: str, run: int) -> np.ndarray:
    m = sio.loadmat(e.ROOT / "Stimuli" / "Envelopes" / story / f"{story}_{run}_env.mat")
    env = np.asarray(m["envelope"], float).ravel()
    fs = float(np.squeeze(m["fsEnv"]))
    return cnd.band_64(env[None], fs, 1.0, 4.0)[0][int(e.START_S * 64):]


def tracking_one(sub: int, data: dict, envs: dict) -> dict:
    runs = [r for r in sorted(data) if sub in data[r]]
    lag = lambda x, T: np.column_stack([np.r_[np.zeros(k), x[: T - k]] for k in range(N_LAGS)])
    out = {}
    for which in ("attended", "ignored"):
        story = e.GROUP[sub] if which == "attended" else ("Journey" if e.GROUP[sub] == "20000" else "20000")
        A, B = {}, {}
        for r in runs:
            Y, v = data[r][sub]
            T = min(Y.shape[1], envs[(story, r)].size)
            A[r], B[r] = lag(envs[(story, r)][:T], T)[v[:T]], Y[:, :T][:, v[:T]].T
        rs = []
        for r in runs:
            X = np.vstack([A[q] for q in runs if q != r])
            Z = np.vstack([B[q] for q in runs if q != r])
            G = X.T @ X
            W = np.linalg.solve(G + np.trace(G) / N_LAGS * np.eye(N_LAGS), X.T @ Z)
            P, Yr = A[r] @ W, B[r]
            P, Yr = P - P.mean(0), Yr - Yr.mean(0)
            rs.append(float(np.mean((P * Yr).sum(0) / np.sqrt((P * P).sum(0) * (Yr * Yr).sum(0) + 1e-300))))
        out[which] = float(np.mean(rs))
    return {"listener": sub, **out}


def decode_one(sub: int, data: dict, envs: dict, L: int = 17) -> dict:
    runs = [r for r in sorted(data) if sub in data[r]]
    att = e.GROUP[sub]
    ign = "Journey" if att == "20000" else "20000"

    def lag_x(Y, T):                      # EEG at t + k predicts the envelope at t
        X = np.zeros((T, Y.shape[0] * L))
        for k in range(L):
            X[: T - k, k::L] = Y[:, k:T].T
        return X
    X, ya, yi = {}, {}, {}
    for r in runs:
        Y, v = data[r][sub]
        T = min(Y.shape[1], envs[(att, r)].size, envs[(ign, r)].size)
        X[r], ya[r], yi[r] = lag_x(np.where(v[:T], Y[:, :T], 0.0), T), envs[(att, r)][:T], envs[(ign, r)][:T]
    G = {r: X[r].T @ X[r] for r in runs}
    c = {r: X[r].T @ ya[r] for r in runs}
    Gt, ct = sum(G.values()), sum(c.values())
    hits, diff = [], []
    for r in runs:
        Gr = Gt - G[r]
        w = np.linalg.solve(Gr + 10 * np.trace(Gr) / Gr.shape[0] * np.eye(Gr.shape[0]), ct - c[r])
        rec = X[r] @ w
        ra, ri = np.corrcoef(rec, ya[r])[0, 1], np.corrcoef(rec, yi[r])[0, 1]
        hits.append(ra > ri)
        diff.append(ra - ri)
    return {"listener": sub, "accuracy": float(np.mean(hits)), "r_att_minus_ign": float(np.mean(diff)), "n_runs": len(runs)}


def attention_checks():
    paths, cfg = load_paths(), load_analysis()
    out = paths.results_root
    data = e.load_all(paths.derived_root / "cocktail_party")
    envs = {(s, r): envelope(s, r) for s in ("20000", "Journey") for r in e.RUNS}
    trk = pd.DataFrame(Parallel(n_jobs=8)(delayed(tracking_one)(s, data, envs) for s in e.SUBJECTS))
    trk.to_csv(out / "attention_tracking.csv", index=False)
    res = {"a_tracking_attended_minus_ignored": {k: v for k, v in summarise((trk.attended - trk.ignored).to_numpy(), "greater").items()
                                                 if k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")},
           "a_mean_r": [float(trk.attended.mean()), float(trk.ignored.mean())]}
    # within-dataset spaces, leave-one-run-out, groups pooled with equal size
    g = np.random.default_rng([cfg.project.seed, 16])
    terms = {}
    for r, d in data.items():
        subs = sorted(d)
        A_, B_ = [s for s in subs if e.GROUP[s] == "20000"], [s for s in subs if e.GROUP[s] == "Journey"]
        k = min(len(A_), len(B_))
        pick = sorted(A_[:k] + B_[:k])
        n = min(d[s][0].shape[1] for s in pick)
        terms[r] = _cov_terms(np.stack([d[s][0][:, :n] for s in pick]), np.stack([d[s][1][:n] for s in pick]))
    rows = []
    for r in sorted(data):
        if len(data[r]) < 8:
            continue
        W = corrca_from_terms(sum(t[0] for q, t in terms.items() if q != r), sum(t[1] for q, t in terms.items() if q != r), 16, 0.5, 3)[0]
        for i, v in e.run_isc(data[r], W, g).items():
            rows.append({"run": r, "listener": i, **v})
    df = pd.DataFrame(rows)
    df.to_csv(out / "attention_isc_within_space.csv", index=False)
    per_run = df.groupby("run").apply(lambda t: (t.same - t.other).mean(), include_groups=False)
    res["b_within_space_runs_same_minus_other"] = {k: v for k, v in summarise(per_run.to_numpy(), "greater").items()
                                                   if k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}
    res["b_mean_isc_same_other"] = [float(df.same.mean()), float(df["other"].mean())]
    dec = pd.DataFrame(Parallel(n_jobs=4)(delayed(decode_one)(s, data, envs) for s in e.SUBJECTS))
    dec.to_csv(out / "attention_decoding.csv", index=False)
    res["c_decoding_accuracy"] = {"mean": float(dec.accuracy.mean()), "median": float(dec.accuracy.median()),
                                  "group_1_17": float(dec.accuracy[dec.listener <= 17].mean()),
                                  "group_18_33": float(dec.accuracy[dec.listener > 17].mean())}
    res["c_r_att_minus_ign"] = {k: v for k, v in summarise(dec.r_att_minus_ign.to_numpy(), "greater").items()
                                if k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}
    (out / "attention_cocktail_checks.json").write_text(json.dumps(res, indent=2, default=float))
