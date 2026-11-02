"""Cross-validated representational geometry of network states.

Crossnobis distances (Walther et al., 2016) are computed within listener from trial-level patterns, with one trial of
each condition forming a partition. The noise covariance is the within-condition residual covariance shrunk towards its
diagonal (Schaefer & Strimmer, 2005) and is estimated without the four trials entering each product (Guggenmos et al., 2018).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

CONDITIONS = ("MC", "MD", "MF", "SD", "SF")


def shrinkage_diag_covariance(residuals: np.ndarray, dof: int) -> tuple[np.ndarray, float]:
    """Schaefer-Strimmer (2005) target-D shrinkage; returns (covariance, lambda)."""
    X = residuals
    s_sum = X.T @ X
    s = s_sum / dof
    var = np.diag(s)
    std = np.sqrt(var)
    X2 = X**2
    s_mean = s_sum / np.outer(std, std) / (X.shape[0] - 1)
    s2_mean = (X2.T @ X2) / np.outer(var, var) / (X.shape[0] - 1)
    var_hat = X.shape[0] / dof**2 * (s2_mean - s_mean**2)
    off = ~np.eye(s.shape[0], dtype=bool)
    lam = float(np.clip(var_hat[off].sum() / (s_mean[off] ** 2).sum(), 0.0, 1.0))
    return s * (np.eye(s.shape[0]) + (1 - lam) * off), lam


class ShrinkagePrecision:
    """Implicit inverse of the Schaefer-Strimmer target-D shrinkage covariance.

    Sigma = lam * diag(s) + (1 - lam) * R'R / dof has a rank-n update of a diagonal, so
    Sigma^-1 v is evaluated with the Woodbury identity in O(nP + n^3); the shrinkage intensity is
    computed from n x n Gram matrices. Numerically identical to inverting the explicit matrix.
    """

    def __init__(self, residuals: np.ndarray, dof: int):
        R = residuals
        n, P = R.shape
        var = (R**2).sum(0) / dof
        Z = R / np.sqrt(var)
        G = Z @ Z.T
        diag_s = (Z**2).sum(0) / (n - 1)
        diag_s2 = (Z**4).sum(0) / (n - 1)
        sum_s = (G**2).sum() / (n - 1) ** 2
        sum_s2 = ((Z**2).sum(1) ** 2).sum() / (n - 1)
        off_s = sum_s - (diag_s**2).sum()
        off_var_hat = n / dof**2 * ((sum_s2 - diag_s2.sum()) - off_s)
        self.lam = float(np.clip(off_var_hat / off_s, 0.0, 1.0)) if off_s > 0 else 1.0
        self.P = P
        if self.lam < 1e-10 or P <= n:
            cov = (1 - self.lam) * (R.T @ R) / dof + self.lam * np.diag(var)
            self._inv = np.linalg.inv(cov)
        else:
            self._inv = None
            self.dinv = 1.0 / (self.lam * var)
            self.U = R * np.sqrt((1 - self.lam) / dof)
            M = np.eye(n) + (self.U * self.dinv) @ self.U.T
            self.Minv = np.linalg.inv(M)

    def solve(self, v: np.ndarray) -> np.ndarray:
        if self._inv is not None:
            return self._inv @ v
        dv = self.dinv * v
        return dv - self.dinv * (self.U.T @ (self.Minv @ (self.U @ dv)))

    def quad(self, u: np.ndarray, v: np.ndarray) -> float:
        return float(u @ self.solve(v))

    def matrix(self) -> np.ndarray:
        return self._inv if self._inv is not None else self.solve(np.eye(self.P))


def _residuals(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    resid = X.copy()
    for c in np.unique(labels):
        m = labels == c
        resid[m] -= X[m].mean(axis=0)
    return resid


def noise_precision(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Precision matrix from within-condition residuals of trial patterns X (trials x features)."""
    conds = np.unique(labels)
    resid = X.copy()
    for c in conds:
        m = labels == c
        resid[m] -= X[m].mean(axis=0)
    cov, lam = shrinkage_diag_covariance(resid, dof=X.shape[0] - conds.size)
    return np.linalg.inv(cov), lam


def _design(Z: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(Z.shape[0]), Z])


def _within_slope(X: np.ndarray, Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Pooled within-condition regression slope of X on Z (k x P)."""
    return np.linalg.lstsq(_residuals(Z, labels), _residuals(X, labels), rcond=None)[0]


def _precision_excluding(X: np.ndarray, labels: np.ndarray, drop: np.ndarray,
                         covariates: np.ndarray | None = None):
    """Noise precision estimated without the trials in ``drop``.

    With covariates, the pooled within-condition slope of the features on the covariates is
    estimated on the kept trials (the ANCOVA adjustment: comprehension is constant within a
    condition, so within-condition covariation reflects acoustic tracking only). Besides the
    full-sample slope (used for the noise residuals), slopes from two disjoint halves of the kept
    trials are returned: the two difference vectors of a crossnobis product are adjusted with
    different halves, so slope error cannot enter the product quadratically.
    """
    keep = np.flatnonzero(~np.isin(np.arange(X.shape[0]), drop))
    betas = None
    Xk = X[keep]
    if covariates is not None:
        beta = _within_slope(Xk, covariates[keep], labels[keep])
        Xk = Xk - covariates[keep] @ beta
        h1, h2 = keep[0::2], keep[1::2]
        betas = tuple(_within_slope(X[h], covariates[h], labels[h]) for h in (h1, h2))
    lab = labels[keep]
    return ShrinkagePrecision(_residuals(Xk, lab), dof=Xk.shape[0] - np.unique(lab).size), betas


def _adjust(x: np.ndarray, z: np.ndarray | None, beta: np.ndarray | None) -> np.ndarray:
    if beta is None:
        return x
    return x - z @ beta


def _diff(X, Z, beta, i, j):
    zi = None if Z is None else Z[i]
    zj = None if Z is None else Z[j]
    return _adjust(X[i], zi, beta) - _adjust(X[j], zj, beta)


def crossnobis_pair(X: np.ndarray, labels: np.ndarray, ci: str, cj: str,
                    covariates: np.ndarray | None = None) -> float:
    I, J = np.flatnonzero(labels == ci), np.flatnonzero(labels == cj)
    if I.size < 2 or J.size < 2:
        raise ValueError("crossnobis requires at least two trials per condition")
    total, count = 0.0, 0
    Z = covariates
    for a, a2 in itertools.combinations(I, 2):
        for b, b2 in itertools.combinations(J, 2):
            quad = np.array([a, a2, b, b2])
            prec, betas = _precision_excluding(X, labels, quad, Z)
            assignments = [(None, None)] if betas is None else [betas, betas[::-1]]
            for b1, b2_ in assignments:
                for u, v in (((a, b), (a2, b2)), ((a, b2), (a2, b))):
                    total += prec.quad(_diff(X, Z, b1, *u), _diff(X, Z, b2_, *v))
                    count += 1
    return total / count / X.shape[1]


def crossnobis_rdm(X: np.ndarray, labels: np.ndarray, conditions=CONDITIONS,
                   covariates: np.ndarray | None = None) -> np.ndarray:
    """5 x 5 crossnobis RDM with cross-validated noise covariance (normalised by the number of features).

    ``covariates`` (trials x k) are regressed out of the features with coefficients estimated,
    for every product, on the same trials as the noise covariance (cross-validated adjustment).
    """
    D = np.zeros((len(conditions), len(conditions)))
    for (a, ca), (b, cb) in itertools.combinations(enumerate(conditions), 2):
        D[a, b] = D[b, a] = crossnobis_pair(X, labels, ca, cb, covariates)
    return D


def lci(D: np.ndarray, conditions=CONDITIONS) -> float:
    """Language Capture Index: [d(MD,MC) - d(MD,SD)] - [d(MF,MC) - d(MF,SF)]."""
    k = {c: i for i, c in enumerate(conditions)}
    d = lambda a, b: D[k[a], k[b]]
    return (d("MD", "MC") - d("MD", "SD")) - (d("MF", "MC") - d("MF", "SF"))


def lci_terms(D: np.ndarray, conditions=CONDITIONS) -> dict[str, float]:
    k = {c: i for i, c in enumerate(conditions)}
    d = lambda a, b: float(D[k[a], k[b]])
    return {"MD_toward_speech": d("MD", "MC") - d("MD", "SD"), "MF_toward_speech": d("MF", "MC") - d("MF", "SF"),
            "d_MD_MC": d("MD", "MC"), "d_MD_SD": d("MD", "SD"), "d_MF_MC": d("MF", "MC"), "d_MF_SF": d("MF", "SF")}


def _half_splits(n: int) -> list[np.ndarray]:
    """All ways of choosing the first half (size floor(n/2)) of n trials."""
    return [np.array(c) for c in itertools.combinations(range(n), n // 2)]


def comprehension_alignment(X: np.ndarray, labels: np.ndarray, max_splits: int | None = None,
                            rng: np.random.Generator | None = None, covariates: np.ndarray | None = None) -> float:
    """Cross-validated cosine between v(speech)=SD-SF and v(song)=MD-MF (the alignment test).

    Trials of each condition are split into halves A and B; the speech vector is estimated on one
    half and the song vector on the other (both assignments). Vectors are whitened with a noise
    covariance estimated from the complementary trials only (the other halves and MC), so the two
    vectors and the whitener are mutually independent and the cosine is zero in expectation under
    the null. The cosine is averaged over all split combinations (or ``max_splits`` random ones).
    """
    idx = {c: np.flatnonzero(labels == c) for c in ("MD", "MF", "SD", "SF")}
    splits = {c: _half_splits(v.size) for c, v in idx.items()}
    combos = list(itertools.product(*(range(len(splits[c])) for c in ("MD", "MF", "SD", "SF"))))
    if max_splits is not None and len(combos) > max_splits:
        rng = rng or np.random.default_rng(0)
        combos = [combos[i] for i in rng.choice(len(combos), max_splits, replace=False)]

    def members(c, s, first):
        sel = np.zeros(idx[c].size, dtype=bool)
        sel[splits[c][s]] = True
        return idx[c][sel if first else ~sel]

    cos = []
    for sm, sf, ssd, ssf in combos:
        for speech_first in (True, False):
            sp = [members("SD", ssd, speech_first), members("SF", ssf, speech_first)]
            so = [members("MD", sm, not speech_first), members("MF", sf, not speech_first)]
            used = np.concatenate(sp + so)
            prec, betas = _precision_excluding(X, labels, used, covariates)
            for b1, b2 in ([(None, None)] if betas is None else [betas, betas[::-1]]):
                Xu = X if b1 is None else X - covariates @ b1
                Xv = X if b2 is None else X - covariates @ b2
                u = Xu[sp[0]].mean(axis=0) - Xu[sp[1]].mean(axis=0)
                v = Xv[so[0]].mean(axis=0) - Xv[so[1]].mean(axis=0)
                su, sv = prec.solve(u), prec.solve(v)
                cos.append(u @ sv / np.sqrt((u @ su) * (v @ sv)))
    return float(np.mean(cos))


def cross_decoding_auc(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Shrinkage-LDA cross-decoding of comprehension: train SD vs SF, test MD vs MF, and vice versa."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def run(train, test):
        tr = np.isin(labels, train)
        te = np.isin(labels, test)
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        clf.fit(X[tr], labels[tr] == train[0])
        return float(roc_auc_score(labels[te] == test[0], clf.decision_function(X[te])))

    s2m, m2s = run(("SD", "SF"), ("MD", "MF")), run(("MD", "MF"), ("SD", "SF"))
    return {"speech_to_song": s2m, "song_to_speech": m2s, "mean": (s2m + m2s) / 2}


@dataclass
class GeometryResult:
    rdm: np.ndarray
    lci: float
    terms: dict
    alignment: float
    shrinkage: float
    n_features: int
    n_trials: dict


def participant_geometry(X: np.ndarray, labels: np.ndarray, max_splits: int | None = None,
                         rng: np.random.Generator | None = None, covariates: np.ndarray | None = None) -> GeometryResult:
    _, lam = noise_precision(X, labels)
    D = crossnobis_rdm(X, labels, covariates=covariates)
    return GeometryResult(rdm=D, lci=lci(D), terms=lci_terms(D),
                          alignment=comprehension_alignment(X, labels, max_splits=max_splits, rng=rng,
                                                            covariates=covariates),
                          shrinkage=lam, n_features=X.shape[1],
                          n_trials={c: int((labels == c).sum()) for c in CONDITIONS})


def crossnobis_halves(means: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
                      residuals: tuple[np.ndarray, np.ndarray], dofs: tuple[int, int],
                      conditions=CONDITIONS) -> np.ndarray:
    """Crossnobis RDM between two independent partitions (Walther et al., 2016, K = 2).

    Used when each partition's condition estimates share a fitted model (VARX cross-fitting):
    partition means m^1, m^2 are independent, and each product is whitened with the noise
    covariance of one partition only, so that E[d] = 0 when conditions do not differ.
    """
    precs = []
    for R, dof in zip(residuals, dofs):
        cov, _ = shrinkage_diag_covariance(R, dof)
        precs.append(np.linalg.inv(cov))
    m1, m2 = means
    P = next(iter(m1.values())).size
    D = np.zeros((len(conditions), len(conditions)))
    for (a, ca), (b, cb) in itertools.combinations(enumerate(conditions), 2):
        u, v = m1[ca] - m1[cb], m2[ca] - m2[cb]
        D[a, b] = D[b, a] = np.mean([u @ Pm @ v for Pm in precs]) / P
    return D


def lci_jackknife_se(X: np.ndarray, labels: np.ndarray, covariates: np.ndarray | None = None) -> float:
    """Leave-one-trial-out jackknife standard error of a participant's LCI."""
    vals = []
    for i in range(X.shape[0]):
        keep = np.arange(X.shape[0]) != i
        lab = labels[keep]
        if min((lab == c).sum() for c in CONDITIONS) < 2:
            continue
        vals.append(lci(crossnobis_rdm(X[keep], lab, covariates=None if covariates is None else covariates[keep])))
    vals = np.asarray(vals)
    n = vals.size
    return float(np.sqrt((n - 1) / n * np.sum((vals - vals.mean()) ** 2)))
