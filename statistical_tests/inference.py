"""Sign-flip tests, effect sizes with exact noncentral-t intervals, equivalence tests, JZS Bayes factors and multiplicity corrections."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

N_PERM = 100_000


def _sign_matrix(n: int) -> np.ndarray:
    codes = np.arange(2**n, dtype=np.int64)[:, None]
    bits = (codes >> np.arange(n)) & 1
    return 1 - 2 * bits.astype(np.int8)


def sign_flip_test(x: np.ndarray, alternative: str = "greater", n_perm: int | None = None,
                   rng: np.random.Generator | None = None, exact_max_n: int = 22) -> tuple[float, np.ndarray]:
    """Sign-flip permutation test of mean(x) against zero; returns (p, null distribution of means)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n <= exact_max_n and n_perm is None:
        null = (_sign_matrix(n) @ x) / n
    else:
        rng = rng or np.random.default_rng(0)
        signs = rng.choice([-1.0, 1.0], size=(n_perm or 10000, n))
        signs[0] = 1.0
        null = signs @ x / n
    obs = x.mean()
    if alternative == "greater":
        p = np.mean(null >= obs - 1e-12)
    elif alternative == "less":
        p = np.mean(null <= obs + 1e-12)
    else:
        p = np.mean(np.abs(null) >= abs(obs) - 1e-12)
    return float(p), null


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=pvals.get)
    m = len(keys)
    adj, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def cohens_dz(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.mean() / x.std(ddof=1))


def dz_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for dz from the noncentral t distribution."""
    return dz_ci_from(cohens_dz(x), len(x), level)


def dz_ci_from(dz: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """Noncentral-t confidence interval for dz given the estimate and the sample size."""
    from scipy.optimize import brentq

    t_obs = dz * np.sqrt(n)
    a = (1 - level) / 2
    def f(nc, q):
        # scipy's nct.cdf returns NaN where the CDF is within machine precision of 1 (noncentrality far below t, of
        # the opposite sign); the CDF decreases in nc, so the limit is 1 below t and 0 above. Without this, large
        v = stats.nct.cdf(t_obs, n - 1, nc)
        return (v if np.isfinite(v) else (1.0 if nc < t_obs else 0.0)) - q

    # bracket: 8 large-sample SDs of the noncentrality estimate on each side
    width = 8 * np.sqrt(1 + t_obs**2 / (2 * (n - 1)))
    try:
        lo = brentq(f, t_obs - width, t_obs + width, args=(1 - a,))
        hi = brentq(f, t_obs - width, t_obs + width, args=(a,))
        return lo / np.sqrt(n), hi / np.sqrt(n)
    except ValueError:
        se = np.sqrt(1 / n + dz**2 / (2 * n))
        z = stats.norm.ppf(1 - a)
        return dz - z * se, dz + z * se


@dataclass
class TostResult:
    bound_dz: float
    p_lower: float
    p_upper: float

    @property
    def p(self) -> float:
        return max(self.p_lower, self.p_upper)


def tost_one_sample(x: np.ndarray, bound_dz: float) -> TostResult:
    """Two one-sided tests of |mean| < bound_dz * SD (Lakens, 2017)."""
    x = np.asarray(x, dtype=float)
    n, m, sd = x.size, x.mean(), x.std(ddof=1)
    se = sd / np.sqrt(n)
    delta = bound_dz * sd
    p_lo = 1 - stats.t.cdf((m + delta) / se, n - 1)
    p_hi = stats.t.cdf((m - delta) / se, n - 1)
    return TostResult(bound_dz=bound_dz, p_lower=float(p_lo), p_upper=float(p_hi))


def sesoi_resource_based(n: int, power: float = 0.8, alpha: float = 0.05, alternative: str = "larger") -> float:
    """dz detectable with the given power for a one-sample t test (Lakens, Scheel & Isager, 2018)."""
    from statsmodels.stats.power import TTestPower

    return float(TTestPower().solve_power(nobs=n, power=power, alpha=alpha, alternative=alternative))


def summarise(x: np.ndarray, alternative: str = "greater", bound_dz: float | None = None) -> dict:
    x = np.asarray(x, dtype=float)
    p, _ = sign_flip_test(x, alternative)
    out = {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
           "sem": float(x.std(ddof=1) / np.sqrt(x.size)), "dz": cohens_dz(x), "dz_ci95": dz_ci(x),
           "p_signflip": p, "n_positive": int((x > 0).sum())}
    if bound_dz is not None:
        t = tost_one_sample(x, bound_dz)
        out.update({"tost_bound_dz": bound_dz, "p_tost": t.p})
    return out


def paired_dz_difference(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, rng: np.random.Generator | None = None,
                         alternative: str = "greater") -> dict:
    """Bootstrap test of dz(a) - dz(b) for paired participant-level estimates (resampling participants)."""
    rng = rng or np.random.default_rng(0)
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = a.size
    obs = cohens_dz(a) - cohens_dz(b)
    idx = rng.integers(0, n, size=(n_boot, n))
    A, B = a[idx], b[idx]
    boot = A.mean(1) / A.std(1, ddof=1) - B.mean(1) / B.std(1, ddof=1)
    centred = boot - boot.mean()
    if alternative == "greater":
        p = float(np.mean(centred >= obs))
    else:
        p = float(np.mean(np.abs(centred) >= abs(obs)))
    return {"dz_difference": float(obs), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p_bootstrap": max(p, 1.0 / n_boot)}


def jzs_bf10(x: np.ndarray, r: float = np.sqrt(2) / 2, alternative: str = "greater") -> float:
    """JZS Bayes factor for a one-sample t test (Rouder et al., 2009): Cauchy(0, r) prior on dz,
    truncated to the tested direction for one-sided alternatives (Morey & Wagenmakers, 2014)."""
    from scipy import integrate, stats

    x = np.asarray(x, dtype=float)
    n = x.size
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    df = n - 1
    like = lambda d: stats.nct.pdf(t, df, d * np.sqrt(n))
    prior = lambda d: stats.cauchy.pdf(d, 0, r)
    lo, hi, mass = {"greater": (0, np.inf, 0.5), "less": (-np.inf, 0, 0.5), "two-sided": (-np.inf, np.inf, 1.0)}[alternative]
    num, _ = integrate.quad(lambda d: like(d) * prior(d), lo, hi, limit=200)
    return float(num / mass / stats.t.pdf(t, df))


def partial_test(x, y, Z, seed):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Zr = np.column_stack([np.ones(len(rx))] + [stats.rankdata(z) for z in Z])
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1])
    g = np.random.default_rng(seed)
    null = np.array([np.corrcoef(ex, g.permutation(ey))[0, 1] for _ in range(N_PERM)])
    return r, float((1 + (null >= r).sum()) / (1 + N_PERM))


def bh(p: np.ndarray) -> np.ndarray:
    o = np.argsort(p)
    m = len(p)
    adj = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(adj, 1.0)
    return out


def pooled_rho(rs, ns):
    z = np.arctanh(np.asarray(rs, float))
    w = (np.asarray(ns, float) - 3) / 1.06
    zp, se = float((w * z).sum() / w.sum()), float(1 / np.sqrt(w.sum()))
    return {"rho": float(np.tanh(zp)), "ci95": [float(np.tanh(zp - 1.96 * se)), float(np.tanh(zp + 1.96 * se))],
            "p_two_sided": float(2 * stats.norm.sf(abs(zp / se)))}


def fisher_ci_unit_variance(r: float, n: int) -> list[float]:
    z, se = np.arctanh(r), np.sqrt(1 / (n - 3))
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]
