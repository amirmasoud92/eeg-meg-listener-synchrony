"""Sufficient-statistics path of the GPU encoding model, used for the two-stream MEG design.

Same estimator as the batched predictions (Schur complement over the per-fold eigendecomposition of the base block,
ridge scaled by the mean eigenvalue of the full training Gram matrix per version), without forming the held-out
predictions.
"""
from __future__ import annotations

import numpy as np
import torch

from encoding_models import gpu_encoding as WG

F64 = torch.float64


class MaskStats:
    """Per-fold masked products of the base block for one mask list (positions among each chapter's valid samples)."""

    def __init__(self, ff: WG.GpuFolds, masks: list[np.ndarray]):
        self.idx, self.m, self.G, self.H, self.a, self.n, self.sy, self.syy = [], [], [], [], [], [], [], []
        for fd, Yv, v, mask in zip(ff.folds, ff.Yv, ff.vnp, masks):
            mask = np.asarray(mask, bool)
            assert not (mask & ~v).any()
            sel = torch.as_tensor(np.flatnonzero(mask[v]), device=ff.dev)
            mv = torch.zeros(int(v.sum()), dtype=F64, device=ff.dev)
            mv[sel] = 1.0
            Xm = fd["XaUv"][sel]
            Ym = Yv[sel]
            self.idx.append(sel)
            self.m.append(mv)
            self.G.append(Xm.T @ Xm)
            self.H.append(Xm.T @ Ym)
            self.a.append(Xm.sum(0))
            self.n.append(float(sel.numel()))
            self.sy.append(Ym.sum(0))
            self.syy.append((Ym * Ym).sum(0))


class SuffTest:
    """One tested stream: versions of its regressor (1 row, unlagged, per chapter) with the other stream's fixed
    regressor; returns per chapter n, sum Y, sum P, sum Y^2, sum P^2, sum YP (K x chapters x 6 x channels)."""

    def __init__(self, ff: WG.GpuFolds, fixed_other: list[np.ndarray], ms: MaskStats, n_lags: int):
        self.ff, self.ms, self.nl = ff, ms, n_lags
        self.Xo = [WG.lagged(ff._t(F[None]), n_lags)[0][v] for F, v in zip(fixed_other, ff.v)]          # Tv x po
        self.po = self.Xo[0].shape[1]
        self.Bo, self.Do, self.Co, self.Qo, self.Ro, self.Cmo, self.smo = [], [], [], [], [], [], []
        for fd, Yv, Xo, mv in zip(ff.folds, ff.Yv, self.Xo, ms.m):
            XaX = fd["XaUv"].T @ torch.cat([Xo, Xo * mv[:, None]], 1)
            self.Bo.append(fd["U"] @ XaX[:, : self.po])
            self.Qo.append(XaX[:, self.po:])
            self.Do.append(Xo.T @ Xo)
            self.Co.append(Xo.T @ Yv)
            Xom = Xo * mv[:, None]
            self.Ro.append(Xom.T @ Xom)
            self.Cmo.append(Xom.T @ Yv)
            self.smo.append(Xom.sum(0))

    def stats(self, F_batch: list[list[np.ndarray]]) -> np.ndarray:
        ff, ms, nl, po = self.ff, self.ms, self.nl, self.po
        K, pa, nch = len(F_batch), ff.pa, ff.nch
        B, D, C, Q, R, Cm, sm = [], [], [], [], [], [], []
        for j, (fd, Yv, v, mv) in enumerate(zip(ff.folds, ff.Yv, ff.v, ms.m)):
            Xt = WG.lagged(ff._t(np.stack([F_batch[k][j] for k in range(K)])), nl)[:, v]                  # K x Tv x pt
            pt = Xt.shape[2]
            Xr = Xt.permute(1, 0, 2).reshape(-1, K * pt)
            Xrm = Xr * mv[:, None]
            XaX = fd["XaUv"].T @ torch.cat([Xr, Xrm], 1)                                               # pa x 2 K pt
            Bt_ = (fd["U"] @ XaX[:, : K * pt]).reshape(pa, K, pt).permute(1, 0, 2)
            Qt_ = XaX[:, K * pt:].reshape(pa, K, pt).permute(1, 0, 2)
            B.append(torch.cat([Bt_, self.Bo[j][None].expand(K, -1, -1)], 2))
            Q.append(torch.cat([Qt_, self.Qo[j][None].expand(K, -1, -1)], 2))
            Xo, Xom = self.Xo[j], self.Xo[j] * mv[:, None]
            Xtm = Xt * mv[None, :, None]
            Dtt = torch.bmm(Xt.transpose(1, 2), Xt)
            Dto = torch.einsum("ktp,tq->kpq", Xt, Xo)
            D.append(torch.cat([torch.cat([Dtt, Dto], 2), torch.cat([Dto.transpose(1, 2), self.Do[j][None].expand(K, -1, -1)], 2)], 1))
            Rtt = torch.bmm(Xtm.transpose(1, 2), Xtm)
            Rto = torch.einsum("ktp,tq->kpq", Xtm, Xom)
            R.append(torch.cat([torch.cat([Rtt, Rto], 2), torch.cat([Rto.transpose(1, 2), self.Ro[j][None].expand(K, -1, -1)], 2)], 1))
            C.append(torch.cat([torch.einsum("ktp,tc->kpc", Xt, Yv), self.Co[j][None].expand(K, -1, -1)], 1))
            Cm.append(torch.cat([torch.einsum("ktp,tc->kpc", Xtm, Yv), self.Cmo[j][None].expand(K, -1, -1)], 1))
            sm.append(torch.cat([Xtm.sum(1), self.smo[j][None].expand(K, -1)], 1))
        ps = pt + po
        BT, DT, CT = sum(B), sum(D), sum(C)
        eye = torch.eye(ps, dtype=F64, device=ff.dev)
        out = np.zeros((K, len(ff.folds), 6, nch))
        for j, fd in enumerate(ff.folds):
            Bt, Dt, Ct = BT - B[j], DT - D[j], CT - C[j]
            lam_eff = ff.lam * (fd["trA"] + Dt.diagonal(dim1=1, dim2=2).sum(1)) / (pa + ps)
            d = 1.0 / (fd["e"][None, :] + lam_eff[:, None])
            UtB = (fd["U"].T @ Bt.permute(1, 0, 2).reshape(pa, K * ps)).reshape(pa, K, ps).permute(1, 0, 2)
            KB = d[:, :, None] * UtB
            KCa = d[:, :, None] * fd["UtCa"][None]
            S = Dt + lam_eff[:, None, None] * eye - UtB.transpose(1, 2) @ KB
            ws = torch.linalg.solve(S, Ct - UtB.transpose(1, 2) @ KCa)                                  # K x ps x ch
            wa = KCa - KB @ ws                                                                          # K x pa x ch
            sp = torch.einsum("p,kpc->kc", ms.a[j], wa) + torch.einsum("ks,ksc->kc", sm[j], ws)
            syp = torch.einsum("pc,kpc->kc", ms.H[j], wa) + torch.einsum("ksc,ksc->kc", Cm[j], ws)
            spp = (torch.einsum("kpc,kpc->kc", wa, torch.matmul(ms.G[j], wa))
                   + 2 * torch.einsum("kpc,kpc->kc", wa, Q[j] @ ws)
                   + torch.einsum("ksc,ksc->kc", ws, R[j] @ ws))
            out[:, j, 0] = ms.n[j]
            out[:, j, 1] = ms.sy[j].cpu().numpy()[None]
            out[:, j, 2] = sp.cpu().numpy()
            out[:, j, 3] = ms.syy[j].cpu().numpy()[None]
            out[:, j, 4] = spp.cpu().numpy()
            out[:, j, 5] = syp.cpu().numpy()
        return out


def r_subset(S: np.ndarray, chapters: np.ndarray) -> np.ndarray:
    """Pooled per-channel r over a subset of chapters from sufficient statistics (... x chapters x 6 x ch)."""
    T = S[..., chapters, :, :].sum(-3)
    n, sy, sp, syy, spp, syp = (T[..., i, :] for i in range(6))
    return (syp - sy * sp / n) / np.sqrt((syy - sy * sy / n) * (spp - sp * sp / n) + 1e-300)
