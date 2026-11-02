"""Multivariate lagged ridge encoding model and its word-locked prediction gain."""
from __future__ import annotations

import numpy as np

RIDGE_GRID = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]


def lagged(F: np.ndarray, T: int, n_lags: int, step: int = 2, first: int = 0) -> np.ndarray:
    """Lags first .. first + n_lags - 1 (in steps of `step` samples)."""
    X = np.zeros((T, F.shape[0] * n_lags))
    for k in range(n_lags):
        d = (first + k) * step
        if d < T:
            X[d:, k::n_lags] = F[:, : T - d].T
    return X


def _design(tr: dict, names: list[str], n_lags: int, lag_spec: dict | None) -> np.ndarray:
    """lag_spec: name -> (first lag, last lag) inclusive, for regressors restricted to a lag window."""
    blocks = []
    for nm in names:
        if lag_spec and nm in lag_spec:
            k0, k1 = lag_spec[nm]
            blocks.append(lagged(tr["F"][nm], tr["T"], k1 - k0 + 1, first=k0))
        else:
            blocks.append(lagged(tr["F"][nm], tr["T"], n_lags))
    return np.hstack(blocks)


def fit_predict(trials: list[dict], names: list[str], lam: float | None, n_lags: int, lag_spec: dict | None = None):
    """Held-out predictions (channels x samples per trial) and the ridge used; lam None = choose on all valid samples."""
    Xs = [_design(tr, names, n_lags, lag_spec) for tr in trials]
    Gs = [X[tr["v"]].T @ X[tr["v"]] for X, tr in zip(Xs, trials)]
    Cs = [X[tr["v"]].T @ tr["Y"][:, tr["v"]].T for X, tr in zip(Xs, trials)]
    G, C = sum(Gs), sum(Cs)
    grid = RIDGE_GRID if lam is None else [lam]
    preds = {g: [] for g in grid}
    for j in range(len(trials)):
        e, V = np.linalg.eigh(G - Gs[j])
        VC, XV = V.T @ (C - Cs[j]), Xs[j] @ V
        for g in grid:
            preds[g].append((XV @ (VC / (e + g * e.mean())[:, None])).T)
    if lam is None:
        score = {g: r_mean(trials, preds[g], [tr["v"] for tr in trials], None) for g in grid}
        lam = max(score, key=score.get)
    return preds[lam], lam


def r_mean(trials, preds, masks, chans) -> float:
    Y = np.hstack([tr["Y"][:, m] for tr, m in zip(trials, masks)])
    P = np.hstack([p[:, m] for p, m in zip(preds, masks)])
    if chans is not None:
        Y, P = Y[chans], P[chans]
    Y = Y - Y.mean(1, keepdims=True)
    P = P - P.mean(1, keepdims=True)
    return float(np.mean((Y * P).sum(1) / np.sqrt((Y * Y).sum(1) * (P * P).sum(1) + 1e-300)))


def r_channels(trials, preds, masks) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel correlations: pooled over trials (channels,) and per trial (trials x channels; NaN if < 64 samples)."""
    def corr(Y, P):
        Y = Y - Y.mean(1, keepdims=True)
        P = P - P.mean(1, keepdims=True)
        return (Y * P).sum(1) / np.sqrt((Y * Y).sum(1) * (P * P).sum(1) + 1e-300)
    pooled = corr(np.hstack([tr["Y"][:, m] for tr, m in zip(trials, masks)]), np.hstack([p[:, m] for p, m in zip(preds, masks)]))
    per = np.array([corr(tr["Y"][:, m], p[:, m]) if m.sum() >= 64 else np.full(tr["Y"].shape[0], np.nan)
                    for tr, p, m in zip(trials, preds, masks)])
    return pooled, per


def suff_stats(trials, preds, masks) -> np.ndarray:
    """Per trial and channel: n, sum Y, sum P, sum Y^2, sum P^2, sum YP (trials x 6 x channels), so that pooled
    correlations over any subset of trials can be recomputed exactly."""
    out = []
    for tr, p, m in zip(trials, preds, masks):
        Y, P = tr["Y"][:, m], p[:, m]
        out.append(np.stack([np.full(Y.shape[0], float(m.sum())), Y.sum(1), P.sum(1), (Y * Y).sum(1), (P * P).sum(1), (Y * P).sum(1)]))
    return np.array(out)


def event_masks(trials: list[dict], key: str, win_s: float) -> list[np.ndarray]:
    out = []
    for tr in trials:
        m = np.zeros(tr["T"], bool)
        w = int(round(win_s * 64))
        for i in tr[key]:
            if i + w > 0 and i < tr["T"]:
                m[max(i, 0): min(i + w, tr["T"])] = True
        out.append(m & tr["v"])
    return out


def delta_r(trials: list[dict], base: list[str], test: str, controls: list[dict], variants: dict, n_lags: int,
            lam: float | None = None) -> tuple[dict, float]:
    """controls: list of {trial index: regressor matrix} replacing trials[j]['F'][test]. variants: name ->
    (mask list or None for all valid samples, channel index array or None)."""
    if lam is None:
        _, lam = fit_predict(trials, base, None, n_lags)
    p_true, _ = fit_predict(trials, base + [test], lam, n_lags)
    r_true = {k: r_mean(trials, p_true, m if m is not None else [tr["v"] for tr in trials], ch) for k, (m, ch) in variants.items()}
    r_ctrl = {k: [] for k in variants}
    saved = [tr["F"][test] for tr in trials]
    for ctrl in controls:
        for j, tr in enumerate(trials):
            tr["F"][test] = ctrl[j]
        p, _ = fit_predict(trials, base + [test], lam, n_lags)
        for k, (m, ch) in variants.items():
            r_ctrl[k].append(r_mean(trials, p, m if m is not None else [tr["v"] for tr in trials], ch))
    for j, tr in enumerate(trials):
        tr["F"][test] = saved[j]
    return {k: r_true[k] - float(np.mean(r_ctrl[k])) for k in variants}, lam


def delta_r_detail(trials: list[dict], base: list[str], test: str, controls: list[dict], masks: list[np.ndarray],
                   n_lags: int, lam: float | None = None, lag_spec: dict | None = None) -> tuple[dict, float]:
    """As delta_r for one variant (masks, all channels), with per-trial and per-channel detail:
    'pooled' (the delta_r value), 'per_channel' (pooled over trials), 'per_trial' (channel mean within each trial),
    'suff_true' / 'suff_ctrl' (sufficient statistics, trials x 6 x channels; controls first)."""
    if lam is None:
        _, lam = fit_predict(trials, base, None, n_lags)
    p_true, _ = fit_predict(trials, base + [test], lam, n_lags, lag_spec)
    pool_t, per_t = r_channels(trials, p_true, masks)
    suff_t, suff_c = suff_stats(trials, p_true, masks), []
    pool_c, per_c = [], []
    saved = [tr["F"][test] for tr in trials]
    for ctrl in controls:
        for j, tr in enumerate(trials):
            tr["F"][test] = ctrl[j]
        p, _ = fit_predict(trials, base + [test], lam, n_lags, lag_spec)
        a, b = r_channels(trials, p, masks)
        pool_c.append(a)
        per_c.append(b)
        suff_c.append(suff_stats(trials, p, masks))
    for j, tr in enumerate(trials):
        tr["F"][test] = saved[j]
    d_ch = pool_t - np.mean(pool_c, axis=0)
    d_tr = np.nanmean(per_t - np.mean(per_c, axis=0), axis=1)
    return {"pooled": float(np.mean(d_ch)), "per_channel": d_ch, "per_trial": d_tr, "suff_true": suff_t,
            "suff_ctrl": np.array(suff_c)}, lam
