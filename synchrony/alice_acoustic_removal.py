"""Alice: comprehension after removal of the acoustic response, with partial correlations and the robustness grid."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from common.settings import load_analysis, load_paths
from preprocessing import cnd_recordings as cnd
from synchrony import alice_comprehension as h
from synchrony.correlated_components import _cov_terms, corrca_from_terms, subject_isc

LAG_STEP, N_LAGS = 2, 20
RIDGE_GRID = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, np.inf]


def features() -> dict[int, np.ndarray]:
    out = {}
    for s in h.SEGMENTS:
        A = cnd.band_64(h.acoustic(s), 64.0, 1.0, 4.0)
        out[s] = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-12)
    return out


def lagged(F: np.ndarray, T: int) -> np.ndarray:
    X = np.zeros((T, F.shape[0] * N_LAGS))
    for k in range(N_LAGS):
        d = k * LAG_STEP
        if d < T:
            X[d:, k::N_LAGS] = F[:, : T - d].T
    return X


def listener_removal(sub: str, data: dict, feats: dict) -> dict:
    a = int(h.START_S * 64)
    segs = sorted(data)
    Xs, Ys, Vs, Gs, Cs = {}, {}, {}, {}, {}
    for s in segs:
        Y, v = data[s]
        T = min(Y.shape[1], feats[s].shape[1] - a)
        X = lagged(feats[s][:, a:a + T], T)
        Xs[s], Ys[s], Vs[s] = X, Y[:, :T], v[:T]
        Gs[s], Cs[s] = X[v[:T]].T @ X[v[:T]], X[v[:T]].T @ Y[:, :T][:, v[:T]].T
    G, C = sum(Gs.values()), sum(Cs.values())
    preds = {lam: {} for lam in RIDGE_GRID}
    for s in segs:
        e, V = np.linalg.eigh(G - Gs[s])
        VC, XV = V.T @ (C - Cs[s]), Xs[s] @ V
        for lam in RIDGE_GRID:
            preds[lam][s] = np.zeros_like(Ys[s]) if np.isinf(lam) else (XV @ (VC / (e + lam * e.mean())[:, None])).T
    sse = {lam: sum(float(((Ys[s][:, Vs[s]] - preds[lam][s][:, Vs[s]]) ** 2).sum()) for s in segs) for lam in RIDGE_GRID}
    best = min(sse, key=sse.get)
    fin = [lam for lam in RIDGE_GRID if not np.isinf(lam)]
    best_fin = min(fin, key=sse.get)                     # r_ac needs a non-zero prediction
    Y = np.hstack([Ys[s][:, Vs[s]] for s in segs])
    P = np.hstack([preds[best_fin][s][:, Vs[s]] for s in segs])
    Yc, Pc = Y - Y.mean(1, keepdims=True), P - P.mean(1, keepdims=True)
    r_ac = float(np.mean((Yc * Pc).sum(1) / np.sqrt((Yc * Yc).sum(1) * (Pc * Pc).sum(1) + 1e-300)))
    return {"listener": sub, "r_ac": r_ac, "lambda": float(best), "delta_power_log": float(np.log(Y.var(1).mean())),
            "resid": {s: (Ys[s] - preds[best][s]).astype(np.float32) for s in segs}}


def isc_per_listener(data: dict, ncomp: int) -> pd.Series:
    subs = sorted(data)
    per = {}
    for seg in h.SEGMENTS:
        have = [s for s in subs if seg in data[s]]
        if len(have) < 4:
            continue
        n = min(data[s][seg][0].shape[1] for s in have)
        per[seg] = (have, np.stack([data[s][seg][0][:, :n] for s in have]), np.stack([data[s][seg][1][:n] for s in have]))
    terms = {seg: _cov_terms(x, v) for seg, (_, x, v) in per.items()}
    vals = {s: [] for s in subs}
    for seg, (have, x, v) in per.items():
        W = corrca_from_terms(sum(t[0] for q, t in terms.items() if q != seg), sum(t[1] for q, t in terms.items() if q != seg),
                              len(subs), 0.5, ncomp)[0]
        for s, val in zip(have, subject_isc(x, W, v).sum(1)):
            vals[s].append(val)
    return pd.Series({s: float(np.nanmean(v)) for s, v in vals.items() if v})


def perm_rho(x, y, alt="greater", n=100000, seed=0):
    r = stats.spearmanr(x, y).correlation
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    g = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, g.permutation(ry))[0, 1] for _ in range(n)])
    p = (1 + (null >= r).sum()) / (1 + n) if alt == "greater" else (1 + (np.abs(null) >= abs(r)).sum()) / (1 + n)
    return float(r), float(p)


def partial(x, y, Z):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Zr = np.column_stack([np.ones(len(rx))] + [stats.rankdata(z) for z in Z])
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def comprehension_after_removal():
    paths, cfg = load_paths(), load_analysis()
    out = paths.results_root
    cache = paths.derived_root / "alice"
    data4, _ = h.load_all(cache, "1-4")
    feats = features()
    rows = Parallel(n_jobs=6)(delayed(listener_removal)(s, data4[s], feats) for s in sorted(data4))
    resid = {r["listener"]: {seg: (r["resid"][seg], data4[r["listener"]][seg][1][: r["resid"][seg].shape[1]]) for seg in r["resid"]}
             for r in rows}
    lis = pd.DataFrame([{k: v for k, v in r.items() if k != "resid"} for r in rows]).set_index("listener")
    lis["isc"] = isc_per_listener(data4, 3)
    lis["isc_res"] = isc_per_listener(resid, 3)
    lis["comprehension"] = h.scores().reindex(lis.index)
    lis.to_csv(out / "alice_removal_listeners.csv")
    d = lis.dropna(subset=["comprehension"])
    res = {"n": int(len(d)), "mean_isc_total_resid": [float(lis.isc.mean()), float(lis.isc_res.mean())],
           "mean_r_ac": float(lis.r_ac.mean())}
    res["rho_comprehension_isc_res"], res["p_one_sided"] = perm_rho(d.comprehension, d.isc_res, "greater", seed=cfg.project.seed)
    res["rho_comprehension_r_ac"], res["p_two_sided"] = perm_rho(d.comprehension, d.r_ac, "two", seed=cfg.project.seed)
    res["partial_comprehension_isc_given_rac_power"] = partial(d.comprehension, d.isc, [d.r_ac, d.delta_power_log])
    res["K3b_partial_comprehension_isc_res_given_rac_power"] = partial(d.comprehension, d.isc_res, [d.r_ac, d.delta_power_log])
    res["descr_rho_isc_r_ac"] = float(stats.spearmanr(d.isc, d.r_ac).correlation)
    res["descr_rho_isc_power"] = float(stats.spearmanr(d.isc, d.delta_power_log).correlation)

    # robustness grid (bands x components), max-t over settings by permutation of comprehension
    data8, _ = h.load_all(cache, "1-8")
    grid = {}
    for band, dd in (("1-4", data4), ("1-8", data8)):
        for nc in (1, 3, 5):
            grid[f"{band}|{nc}"] = isc_per_listener(dd, nc).reindex(d.index)
    M = pd.DataFrame(grid)
    comp_r = stats.rankdata(d.comprehension)
    Mr = M.rank().to_numpy()
    obs = np.array([np.corrcoef(comp_r, Mr[:, j])[0, 1] for j in range(Mr.shape[1])])
    g = np.random.default_rng(cfg.project.seed)
    null_max = np.array([np.max([np.corrcoef(p, Mr[:, j])[0, 1] for j in range(Mr.shape[1])])
                         for p in (g.permutation(comp_r) for _ in range(20000))])
    res["R_TH3a_grid"] = {k: {"rho": float(r), "p_fwe": float((1 + (null_max >= r).sum()) / (1 + null_max.size))}
                          for k, r in zip(M.columns, obs)}
    (out / "alice_removal.json").write_text(json.dumps(res, indent=2, default=float))
