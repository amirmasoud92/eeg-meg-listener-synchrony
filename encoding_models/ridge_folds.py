"""Exact leave-one-out ridge folds through sufficient statistics of the base block."""
from __future__ import annotations

import numpy as np

from encoding_models import ridge_encoding as E

class FastFolds:
    """Per-fold factorisation of the base block of a list of trials (ridge_encoding trial format)."""

    def __init__(self, trials: list[dict], base: list[str], n_lags: int, lam: float):
        self.trials, self.lam, self.n_lags = trials, lam, n_lags
        self.Xa = [E._design(tr, base, n_lags, None) for tr in trials]
        self.v = [tr["v"] for tr in trials]
        Ga = [X[v].T @ X[v] for X, v in zip(self.Xa, self.v)]
        Ca = [X[v].T @ tr["Y"][:, v].T for X, v, tr in zip(self.Xa, self.v, trials)]
        GA, CA = sum(Ga), sum(Ca)
        self.pa = GA.shape[0]
        self.folds = []
        for j in range(len(trials)):
            A, Cq = GA - Ga[j], CA - Ca[j]
            e, U = np.linalg.eigh(A)
            self.folds.append({"e": e, "U": U, "trA": float(np.trace(A)), "UtCa": U.T @ Cq, "XaU": self.Xa[j] @ U})

    def predict(self, Xs: list[np.ndarray]) -> list[np.ndarray]:
        """Held-out predictions (channels x samples per trial) with the tested block Xs (samples x p_s per trial)."""
        v = self.v
        B = [Xa[vv].T @ X[vv] for Xa, X, vv in zip(self.Xa, Xs, v)]
        D = [X[vv].T @ X[vv] for X, vv in zip(Xs, v)]
        Cs = [X[vv].T @ tr["Y"][:, vv].T for X, vv, tr in zip(Xs, v, self.trials)]
        BT, DT, CT = sum(B), sum(D), sum(Cs)
        ps = Xs[0].shape[1]
        out = []
        for j, fd in enumerate(self.folds):
            Bt, Dt, Ct = BT - B[j], DT - D[j], CT - Cs[j]
            lam_eff = self.lam * (fd["trA"] + float(np.trace(Dt))) / (self.pa + ps)
            d = 1.0 / (fd["e"] + lam_eff)
            UtB = fd["U"].T @ Bt
            KB_t = d[:, None] * UtB                         # U' K B  (in the eigenbasis)
            KCa_t = d[:, None] * fd["UtCa"]                 # U' K C_a
            S = Dt + lam_eff * np.eye(ps) - UtB.T @ KB_t
            ws = np.linalg.solve(S, Ct - UtB.T @ KCa_t)
            wa_t = KCa_t - KB_t @ ws                        # U' w_a
            out.append((fd["XaU"] @ wa_t + Xs[j] @ ws).T)
        return out


def _test_design(trials, test, n_lags, lag_spec):
    return [E._design(tr, [test], n_lags, lag_spec) for tr in trials]


def delta_r_detail(trials: list[dict], base: list[str], test: str, controls: list[dict], masks: list[np.ndarray],
                   n_lags: int, lam: float | None = None, lag_spec: dict | None = None, folds: FastFolds | None = None):
    """Same output as ridge_encoding.delta_r_detail, plus 'null_pooled' (per-control pooled channel-mean r) for the
    Monte Carlo SE of the null mean."""
    if lam is None:
        _, lam = E.fit_predict(trials, base, None, n_lags)
    folds = folds or FastFolds(trials, base, n_lags, lam)
    p_true = folds.predict(_test_design(trials, test, n_lags, lag_spec))
    pool_t, per_t = E.r_channels(trials, p_true, masks)
    suff_t, suff_c, pool_c, per_c = E.suff_stats(trials, p_true, masks), [], [], []
    saved = [tr["F"][test] for tr in trials]
    for ctrl in controls:
        for j, tr in enumerate(trials):
            tr["F"][test] = ctrl[j]
        p = folds.predict(_test_design(trials, test, n_lags, lag_spec))
        a, b_ = E.r_channels(trials, p, masks)
        pool_c.append(a)
        per_c.append(b_)
        suff_c.append(E.suff_stats(trials, p, masks))
    for j, tr in enumerate(trials):
        tr["F"][test] = saved[j]
    d_ch = pool_t - np.mean(pool_c, axis=0)
    d_tr = np.nanmean(per_t - np.mean(per_c, axis=0), axis=1)
    return {"pooled": float(np.mean(d_ch)), "per_channel": d_ch, "per_trial": d_tr, "suff_true": suff_t,
            "suff_ctrl": np.array(suff_c), "null_pooled": np.array([float(np.mean(x)) for x in pool_c]),
            "true_pooled": float(np.mean(pool_t))}, lam


def delta_r(trials: list[dict], base: list[str], test: str, controls: list[dict], variants: dict, n_lags: int,
            lam: float | None = None, folds: FastFolds | None = None):
    """Same output as ridge_encoding.delta_r (several metric variants), plus the per-control null values per variant."""
    if lam is None:
        _, lam = E.fit_predict(trials, base, None, n_lags)
    folds = folds or FastFolds(trials, base, n_lags, lam)
    p_true = folds.predict(_test_design(trials, test, n_lags, None))
    allv = [tr["v"] for tr in trials]
    r_true = {k: E.r_mean(trials, p_true, m if m is not None else allv, ch) for k, (m, ch) in variants.items()}
    r_ctrl = {k: [] for k in variants}
    saved = [tr["F"][test] for tr in trials]
    for ctrl in controls:
        for j, tr in enumerate(trials):
            tr["F"][test] = ctrl[j]
        p = folds.predict(_test_design(trials, test, n_lags, None))
        for k, (m, ch) in variants.items():
            r_ctrl[k].append(E.r_mean(trials, p, m if m is not None else allv, ch))
    for j, tr in enumerate(trials):
        tr["F"][test] = saved[j]
    return {k: r_true[k] - float(np.mean(r_ctrl[k])) for k in variants}, lam, {k: np.array(v) for k, v in r_ctrl.items()}
