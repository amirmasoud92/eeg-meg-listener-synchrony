"""Cross-fitted decomposition into envelope-driven and recurrent parts, the index on each part, and Granger influences."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.settings import rng as make_rng
from network_states.network_features import Band, network_edge_means, network_node_means
from network_states.representational_geometry import CONDITIONS, crossnobis_halves, lci, lci_terms
from network_states.varx_trials import VarxTrial, component_features, decompose, equalise_fit_epochs, fit_condition

COMPONENTS = ("raw", "ext", "int")


@dataclass(frozen=True)
class VarxSettings:
    na: int
    nb: int
    lam: float
    sfreq: float
    epoch_s: float
    spectral_epoch_s: float
    n_boot: int
    n_splits: int
    bands: tuple[Band, ...]
    band_idx: tuple[int, ...]

    @property
    def epoch_n(self) -> int:
        return int(round(self.epoch_s * self.sfreq))


def h2_pattern(f: dict, band_idx, membership: np.ndarray) -> np.ndarray:
    parts = []
    for b in band_idx:
        parts += [network_edge_means(f["dwpli"][b], membership), network_edge_means(f["aec"][b], membership),
                  network_node_means(f["logpower"][b], membership)]
    return np.concatenate(parts)


def _features(sig, ok, n_eq, st: VarxSettings, membership, seed, key, tag):
    used = [st.bands[i] for i in st.band_idx]
    f = component_features(sig.T, ok, n_eq, st.sfreq, used, st.epoch_s, st.spectral_epoch_s, st.n_boot,
                           make_rng(seed, "aim2eq", key, tag))
    return h2_pattern(f, range(len(used)), membership)


def granger_and_fits(trials: list[VarxTrial], st: VarxSettings, edges: list[tuple[int, int]],
                     g: np.random.Generator) -> tuple[list[dict], list[dict]]:
    """Full per-condition fits on length-equalised data: edge Granger R and model diagnostics."""
    by_cond = {c: [t for t in trials if t.condition == c] for c in CONDITIONS}
    n_keep = min(sum(int(t.ok.sum()) for t in v) for v in by_cond.values())
    gr, fits = [], []
    for cond, ts in by_cond.items():
        fit = fit_condition(ts, equalise_fit_epochs(ts, n_keep, g), st.na, st.nb, st.lam, st.epoch_n)
        m = fit.model
        off = ~np.eye(m.A.shape[1], dtype=bool)
        fits.append({"condition": cond, "T": m.T, "spectral_radius": fit.spectral_radius,
                     "explained_by_input": fit.explained_by_input, "B_R_mean": float(m.B_R.mean()),
                     "A_R_mean": float(m.A_R[off].mean())})
        for src, dst in edges:
            gr.append({"condition": cond, "edge": f"{src}->{dst}", "R": float(m.A_R[dst, src]),
                       "p": float(m.A_pval[dst, src])})
    return gr, fits


def crossfit_geometry(trials: list[VarxTrial], st: VarxSettings, membership: np.ndarray, seed: int,
                      g: np.random.Generator) -> dict[str, np.ndarray]:
    """Average crossnobis RDM of the raw, extrinsic and intrinsic components over half-splits."""
    by_cond = {c: [t for t in trials if t.condition == c] for c in CONDITIONS}
    n_eq = min(int(t.ok.sum()) for t in trials)
    raw = {t.key: _features(t.y - np.nanmean(t.y, axis=0), t.ok, n_eq, st, membership, seed, t.key, "raw")
           for t in trials}
    D_sum = {c: 0.0 for c in COMPONENTS}
    for r in range(st.n_splits):
        halves = {}
        for c, ts in by_cond.items():
            idx = g.permutation(len(ts))
            k = len(ts) // 2
            halves[c] = ([ts[i] for i in idx[:k]], [ts[i] for i in idx[k:]])
        means = {c: ({}, {}) for c in COMPONENTS}
        resid = {c: ([], []) for c in COMPONENTS}
        for h in range(2):
            part = {c: halves[c][h] for c in CONDITIONS}
            keep = min(sum(int(t.ok.sum()) for t in v) for v in part.values())
            for cond, ts in part.items():
                fit = fit_condition(ts, equalise_fit_epochs(ts, keep, g), st.na, st.nb, st.lam, st.epoch_n,
                                    granger=False, diagnostics=False)
                vecs = {"raw": [], "ext": [], "int": []}
                for t in ts:
                    ext, intr = decompose(fit.model, t.y, t.x, fit.y_mean, fit.x_mean)
                    vecs["raw"].append(raw[t.key])
                    vecs["ext"].append(_features(ext, t.ok, n_eq, st, membership, seed, t.key, f"ext{r}{h}"))
                    vecs["int"].append(_features(intr, t.ok, n_eq, st, membership, seed, t.key, f"int{r}{h}"))
                for comp, v in vecs.items():
                    v = np.array(v)
                    means[comp][h][cond] = v.mean(axis=0)
                    resid[comp][h].append(v - v.mean(axis=0))
        dofs = tuple(max(int(sum(len(halves[c][h]) - 1 for c in CONDITIONS)), 1) for h in range(2))
        for comp in COMPONENTS:
            D_sum[comp] = D_sum[comp] + crossnobis_halves(means[comp], tuple(np.vstack(x) for x in resid[comp]), dofs)
    return {c: D_sum[c] / st.n_splits for c in COMPONENTS}


def lci_rows(rdms: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for comp, D in rdms.items():
        rows.append({"component": comp, "lci": lci(D), **lci_terms(D),
                     **{f"rdm_{a}_{b}": D[i, j] for i, a in enumerate(CONDITIONS) for j, b in enumerate(CONDITIONS)
                        if i < j}})
    return rows
