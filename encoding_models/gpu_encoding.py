"""GPU implementation of the word-level encoding model (float64), with permuted regressors in batches."""
from __future__ import annotations

import itertools
import os

import numpy as np
import torch

from encoding_models import ridge_encoding as E

F64 = torch.float64


def device() -> torch.device:
    return torch.device(os.environ.get("ENCODING_DEVICE", "cuda:0"))


class GpuFolds:
    """Per-fold factorisation of the base block (ridge_encoding trial format); lam None = select it as ridge_encoding does."""

    def __init__(self, trials: list[dict], base: list[str], n_lags: int, lam: float | None = None, dev: torch.device | None = None):
        """Every metric uses samples inside the validity mask (event masks are restricted to it), so only valid rows are
        kept: per fold XaUv = Xa_j[v] U_j (U_j orthogonal), and B_j = Xa_j[v]' Xs_j[v] = U_j XaUv_j' Xs_j[v]."""
        self.dev = dev or device()
        self.trials, self.n_lags = trials, n_lags
        self.vnp = [np.asarray(tr["v"], bool) for tr in trials]
        self.v = [torch.as_tensor(v, device=self.dev) for v in self.vnp]
        self.Yv = [self._t(tr["Y"][:, v].T) for tr, v in zip(trials, self.vnp)]   # valid samples x channels
        self.Yvc = [Y.T.contiguous() for Y in self.Yv]                            # channels x valid samples
        self.nch = self.Yv[0].shape[1]
        self.folds = []
        Ga, Ca = [], []
        for tr, v, Yv in zip(trials, self.vnp, self.Yv):
            Xv = self._t(E._design(tr, base, n_lags, None)[v])
            Ga.append(Xv.T @ Xv)
            Ca.append(Xv.T @ Yv)
        GA, CA = sum(Ga), sum(Ca)
        self.pa = GA.shape[0]
        for j, (tr, v) in enumerate(zip(trials, self.vnp)):
            A = GA - Ga[j]
            e, U = torch.linalg.eigh(A)
            XaUv = self._t(E._design(tr, base, n_lags, None)[v]) @ U
            self.folds.append({"e": e, "U": U, "trA": float(torch.trace(A)), "UtCa": U.T @ (CA - Ca[j]), "XaUv": XaUv})
        self.all_valid = MaskSet(self, [tr["v"] for tr in trials])
        self.lam = self.select_lambda() if lam is None else lam

    def _t(self, x) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(x), dtype=F64, device=self.dev)

    def select_lambda(self) -> float:
        score = {}
        for g in E.RIDGE_GRID:
            preds = [(fd["XaUv"] @ (fd["UtCa"] / (fd["e"] + g * fd["e"].mean())[:, None])).T[None] for fd in self.folds]
            score[g] = float(self.all_valid.stats(preds)["pooled"][0].mean())
        return max(score, key=score.get)

    def predict_batch(self, F_batch: list[list[np.ndarray]], n_lags: int, first: int = 0) -> list[torch.Tensor]:
        """Held-out predictions for K versions of the tested regressor (per version, per trial: rows x samples,
        unlagged), lagged as ridge_encoding.lagged with lags first .. first + n_lags - 1: per trial K x channels x valid samples."""
        K, pa, nch = len(F_batch), self.pa, self.nch
        Xs = [lagged(self._t(np.stack([F_batch[k][j] for k in range(K)])), n_lags, first)[:, v] for j, v in enumerate(self.v)]   # K x Tv x ps
        ps = Xs[0].shape[2]
        B, D, Cs = [], [], []
        for X, fd, Yv in zip(Xs, self.folds, self.Yv):
            Xr = X.permute(1, 0, 2).reshape(-1, K * ps)                                                  # Tv x (K ps)
            B.append((fd["U"] @ (fd["XaUv"].T @ Xr)).reshape(pa, K, ps).permute(1, 0, 2))
            Cs.append((Xr.T @ Yv).reshape(K, ps, nch))
            Xk = Xr.reshape(-1, K, ps).permute(1, 0, 2).contiguous()                                   # K x Tv x ps
            D.append(torch.bmm(Xk.transpose(1, 2), Xk))
        BT, DT, CT = sum(B), sum(D), sum(Cs)
        eye = torch.eye(ps, dtype=F64, device=self.dev)
        out = []
        for j, fd in enumerate(self.folds):
            Bt, Dt, Ct = BT - B[j], DT - D[j], CT - Cs[j]
            lam_eff = self.lam * (fd["trA"] + Dt.diagonal(dim1=1, dim2=2).sum(1)) / (pa + ps)          # K
            d = 1.0 / (fd["e"][None, :] + lam_eff[:, None])                                            # K x pa
            UtB = (fd["U"].T @ Bt.permute(1, 0, 2).reshape(pa, K * ps)).reshape(pa, K, ps).permute(1, 0, 2)
            KB = d[:, :, None] * UtB
            KCa = d[:, :, None] * fd["UtCa"][None]
            S = Dt + lam_eff[:, None, None] * eye - UtB.transpose(1, 2) @ KB
            ws = torch.linalg.solve(S, Ct - UtB.transpose(1, 2) @ KCa)                                  # K x ps x ch
            wa = KCa - KB @ ws                                                                          # K x pa x ch
            P = (fd["XaUv"] @ wa.permute(1, 0, 2).reshape(pa, K * nch)).reshape(-1, K, nch).permute(1, 2, 0)
            out.append(P + (Xs[j] @ ws).transpose(1, 2))
        return out


    def solve_blocks(self, Xs: list[torch.Tensor], B: list, D: list, Cs: list) -> list[torch.Tensor]:
        """The Schur step of predict_batch for precomputed blocks (Xs: per trial K x Tv x ps on the valid samples;
        B: K x pa x ps; D: K x ps x ps; Cs: K x ps x ch). Same arithmetic as predict_batch."""
        K, pa, nch = Xs[0].shape[0], self.pa, self.nch
        ps = Xs[0].shape[2]
        BT, DT, CT = sum(B), sum(D), sum(Cs)
        eye = torch.eye(ps, dtype=F64, device=self.dev)
        out = []
        for j, fd in enumerate(self.folds):
            Bt, Dt, Ct = BT - B[j], DT - D[j], CT - Cs[j]
            lam_eff = self.lam * (fd["trA"] + Dt.diagonal(dim1=1, dim2=2).sum(1)) / (pa + ps)
            d = 1.0 / (fd["e"][None, :] + lam_eff[:, None])
            UtB = (fd["U"].T @ Bt.permute(1, 0, 2).reshape(pa, K * ps)).reshape(pa, K, ps).permute(1, 0, 2)
            KB = d[:, :, None] * UtB
            KCa = d[:, :, None] * fd["UtCa"][None]
            S = Dt + lam_eff[:, None, None] * eye - UtB.transpose(1, 2) @ KB
            ws = torch.linalg.solve(S, Ct - UtB.transpose(1, 2) @ KCa)
            wa = KCa - KB @ ws
            P = (fd["XaUv"] @ wa.permute(1, 0, 2).reshape(pa, K * nch)).reshape(-1, K, nch).permute(1, 2, 0)
            out.append(P + (Xs[j] @ ws).transpose(1, 2))
        return out


def lagged(F: torch.Tensor, n_lags: int, first: int = 0, step: int = 2) -> torch.Tensor:
    """ridge_encoding.lagged for a batch (K x rows x T -> K x T x rows*n_lags; same column order, zeros before each lag)."""
    K, r, T = F.shape
    X = torch.zeros((K, T, r * n_lags), dtype=F.dtype, device=F.device)
    for k in range(n_lags):
        d = (first + k) * step
        if d < T:
            X[:, d:, k::n_lags] = F[:, :, : T - d].transpose(1, 2)
    return X


class MaskSet:
    """Masked samples of each trial (event masks or all valid samples), with the EEG side precomputed."""

    def __init__(self, ff: GpuFolds, masks: list[np.ndarray]):
        self.dev, self.nch = ff.dev, ff.nch        
        idx = []
        for m, v in zip(masks, ff.vnp):
            m = np.asarray(m, bool)
            assert not (m & ~v).any(), "masks must lie within the valid samples"
            idx.append(torch.as_tensor(np.flatnonzero(m[v]), device=ff.dev))     # positions among the valid samples
        self.idx = idx
        self.n = [int(i.numel()) for i in self.idx]
        self.N = sum(self.n)
        Ym = [Y[:, i] for Y, i in zip(ff.Yvc, self.idx)]
        self.ybar = sum(Y.sum(1) for Y in Ym) / self.N
        self.Yg = [Y - self.ybar[:, None] for Y in Ym]
        self.syy = sum((Y * Y).sum(1) for Y in self.Yg)
        self.Yt = [(Y - Y.mean(1, keepdim=True)) if n >= 64 else None for Y, n in zip(Ym, self.n)]
        self.tyy = [(Y * Y).sum(1) if Y is not None else None for Y in self.Yt]
        self.Ym = Ym

    def stats(self, P: list[torch.Tensor], suff: bool = False) -> dict:
        """pooled (K x ch), per trial (K x trials x ch; NaN below 64 samples), optionally suff (K x trials x 6 x ch)."""
        Pm = [p[:, :, i] for p, i in zip(P, self.idx)]
        K = Pm[0].shape[0]
        pbar = sum(p.sum(2) for p in Pm) / self.N
        spp = torch.zeros_like(pbar)
        syp = torch.zeros_like(pbar)
        per = []
        for Yg, Yt, tyy, p in zip(self.Yg, self.Yt, self.tyy, Pm):
            pg = p - pbar[:, :, None]
            spp += (pg * pg).sum(2)
            syp += (Yg[None] * pg).sum(2)
            if Yt is None:
                per.append(torch.full((K, self.nch), float("nan"), dtype=F64, device=self.dev))
            else:
                pt = p - p.mean(2, keepdim=True)
                per.append((Yt[None] * pt).sum(2) / torch.sqrt(tyy[None] * (pt * pt).sum(2) + 1e-300))
        out = {"pooled": (syp / torch.sqrt(self.syy[None] * spp + 1e-300)).cpu().numpy(), "per": torch.stack(per, 1).cpu().numpy()}
        if suff:
            s = []
            for Y, p, n in zip(self.Ym, Pm, self.n):
                s.append(torch.stack([torch.full((K, self.nch), float(n), dtype=F64, device=self.dev), Y.sum(1)[None].expand(K, -1),
                                      p.sum(2), (Y * Y).sum(1)[None].expand(K, -1), (p * p).sum(2), (Y[None] * p).sum(2)], 1))
            out["suff"] = torch.stack(s, 1).cpu().numpy()
        return out


def _lags(n_lags: int, lag_spec: dict | None, test: str) -> tuple[int, int]:
    """(number of lags, first lag) for the tested regressor, as ridge_encoding._design."""
    if lag_spec and test in lag_spec:
        k0, k1 = lag_spec[test]
        return k1 - k0 + 1, k0
    return n_lags, 0


def run_test(folds: GpuFolds, trials, test, controls, msets: dict, n_lags, lag_spec=None, batch: int = 8, suff_for=()) -> tuple[dict, dict]:
    """True and per-control statistics for each mask set. controls: list or iterator of {trial: regressor} (consumed
    in order, so an iterator gives the same draws as a list)."""
    nl, first = _lags(n_lags, lag_spec, test)
    p = folds.predict_batch([[tr["F"][test] for tr in trials]], nl, first)
    true = {k: {kk: vv[0] for kk, vv in ms.stats(p, k in suff_for).items()} for k, ms in msets.items()}
    acc = {k: {} for k in msets}
    it = iter(controls)
    while True:
        chunk = list(itertools.islice(it, batch))
        if not chunk:
            break
        p = folds.predict_batch([[ctrl[j] for j in range(len(trials))] for ctrl in chunk], nl, first)
        for k, ms in msets.items():
            for kk, vv in ms.stats(p, k in suff_for).items():
                acc[k].setdefault(kk, []).append(vv)
    ctrl = {k: {kk: np.concatenate(vv) for kk, vv in d.items()} for k, d in acc.items()}
    return true, ctrl


def channel_mean(true: dict, ctrl: dict, chans=None) -> tuple[float, np.ndarray]:
    """r_mean (mean over channels of the pooled r) for the true model and for each control."""
    ix = slice(None) if chans is None else np.asarray(chans)
    return float(np.mean(true["pooled"][ix])), ctrl["pooled"][:, ix].mean(1)


def delta_r_detail(trials, base, test, controls, masks, n_lags, lam=None, lag_spec=None, folds: GpuFolds | None = None, batch: int = 8):
    folds = folds or GpuFolds(trials, base, n_lags, lam)
    if lam is not None:
        folds.lam = lam
    true, ctrl = run_test(folds, trials, test, controls, {"m": MaskSet(folds, masks)}, n_lags, lag_spec, batch, suff_for=("m",))
    t, c = true["m"], ctrl["m"]
    d_ch = t["pooled"] - c["pooled"].mean(0)
    d_tr = np.nanmean(t["per"] - c["per"].mean(0), axis=1)
    return {"pooled": float(np.mean(d_ch)), "per_channel": d_ch, "per_trial": d_tr, "suff_true": t["suff"], "suff_ctrl": c["suff"],
            "null_pooled": c["pooled"].mean(1), "true_pooled": float(np.mean(t["pooled"]))}, folds.lam


def delta_r(trials, base, test, controls, variants, n_lags, lam=None, folds: GpuFolds | None = None, batch: int = 8):
    folds = folds or GpuFolds(trials, base, n_lags, lam)
    if lam is not None:
        folds.lam = lam
    msets, names = {}, {}
    for k, (m, ch) in variants.items():
        key = id(m) if m is not None else "all"
        if key not in msets:
            msets[key] = folds.all_valid if m is None else MaskSet(folds, m)
        names[k] = (key, ch)
    true, ctrl = run_test(folds, trials, test, controls, msets, n_lags, None, batch)
    out, nulls = {}, {}
    for k, (key, ch) in names.items():
        t, c = channel_mean(true[key], ctrl[key], ch)
        out[k], nulls[k] = t - float(np.mean(c)), c
    return out, folds.lam, nulls
