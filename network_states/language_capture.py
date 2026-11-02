"""Language-capture index, alignment of comprehension axes and cross-decoding in the Aalborg data."""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common.settings import PROJECT_ROOT, load_analysis, load_paths, rng
from network_states.connectivity_measures import pair_indices
from network_states.envelope_statistics import AcousticCovariates, envelope_statistics
from network_states.network_features import BAND_NAMES, band_vector, network_membership, stacked_vector
from network_states.representational_geometry import CONDITIONS, cross_decoding_auc, crossnobis_rdm, lci, lci_jackknife_se, lci_terms, participant_geometry
from readers.aalborg_release import load_processed
from statistical_tests.inference import holm, paired_dz_difference, sesoi_resource_based, summarise

RSCRIPT = "Rscript"


def representations(spec: list[dict]) -> dict[str, dict]:
    return {r["name"]: r for r in spec}


def drop_regions(feat: dict, membership: np.ndarray, region_names: list[str], excluded: list[str]):
    """Remove regions (by base name) from region- and pair-level arrays."""
    keep = np.array([r.split("_")[0] not in excluded for r in region_names])
    i, j = pair_indices(len(region_names))
    pk = keep[i] & keep[j]
    out = {}
    for k, v in feat.items():
        if k in ("dwpli", "aec"):
            out[k] = v[..., pk]
        elif k in ("power", "exponent", "offset", "r_squared", "fit_error"):
            out[k] = v[..., keep]
        else:
            out[k] = v
    return out, membership[keep]


def patterns(feat: dict, rep: dict, membership: np.ndarray, region_names: list[str] | None = None) -> np.ndarray:
    if rep.get("exclude_regions"):
        feat, membership = drop_regions(feat, membership, region_names, rep["exclude_regions"])
    bands = [BAND_NAMES.index(b) for b in rep["bands"]]
    summ = rep["summarised"]
    if len(bands) == 1:
        X = band_vector(feat, bands[0], summ, membership)
    else:
        X = stacked_vector(feat, bands, summ, membership)
    if rep.get("drop_aperiodic"):
        n_ap = 2 * (5 if summ else membership.size)
        X = X[:, :-n_ap]
    return X


def load_features(paths, source: str):
    arr = dict(np.load(paths.derived_root / "network_features" / f"features_{source}.npz", allow_pickle=True))
    meta = pd.read_csv(paths.derived_root / "network_features" / f"features_{source}_meta.csv")
    assert (arr["keys"] == meta.key.values).all()
    return arr, meta


def stimulus_labels(meta: pd.DataFrame) -> list[str]:
    return [f"{e}_{c}." for e, c in zip(meta.excerpt, meta.condition)]


def acoustic_model(paths, cfg) -> AcousticCovariates:
    env = {}
    for f in cfg.dataset.processed_files.values():
        for t in load_processed(paths.processed_dir / f):
            env.setdefault(t.label, t.envelope)
    c = cfg.section("geometry")["acoustic_covariates"]
    return AcousticCovariates(env, cfg.dataset.processed_sfreq, c["pca_variance"], c["welch_window_s"])


def listener_geometry(s, idx, feat, labels, reps, membership, max_splits, seed, covariates=None, region_names=None,
                 extra_covariates=None):
    sub = {k: v[idx] for k, v in feat.items() if k != "keys"}
    lab = labels[idx]
    rows = []
    for name, rep in reps.items():
        X = patterns(sub, rep, membership, region_names)
        cov = None
        if rep.get("acoustic_adjusted"):
            cov = covariates[idx]
        elif rep.get("covariate"):
            cov = extra_covariates[rep["covariate"]][idx][:, None]
        g = participant_geometry(X, lab, max_splits=max_splits, rng=rng(seed, "align", s, name), covariates=cov)
        row = {"subject": s, "representation": name, "lci": g.lci, "alignment": g.alignment,
               "shrinkage": g.shrinkage, "n_features": g.n_features, **g.terms,
               **{f"n_{c}": n for c, n in g.n_trials.items()},
               **{f"rdm_{a}_{b}": g.rdm[i, j] for i, a in enumerate(CONDITIONS) for j, b in enumerate(CONDITIONS) if i < j}}
        if rep.get("cross_decode"):
            row.update({f"auc_{k}": v for k, v in cross_decoding_auc(X, lab).items()})
        if rep.get("jackknife"):
            row["lci_jackknife_se"] = lci_jackknife_se(X, lab)
        rows.append(row)
    return rows


def geometry_tests(source: str, reps_filter: list[str] | None, out_tag: str):
    paths, cfg = load_paths(), load_analysis()
    c5 = cfg.section("geometry")
    feat, meta = load_features(paths, source)
    labels = meta.condition.to_numpy()
    region_names = np.load(paths.derived_root / "source_model" / "filters_S1.npz")["regions"].tolist()
    membership = network_membership(region_names)
    reps = representations(c5["representations"])
    if reps_filter:
        reps = {k: v for k, v in reps.items() if k in reps_filter}
    groups = meta.groupby("subject").indices
    covariates = acoustic_model(paths, cfg)(stimulus_labels(meta))
    qc2 = pd.read_csv(paths.derived_root / "epochs" / "trial_inclusion.csv").set_index("key")
    extra = {"frontal_delta": qc2.loc[meta.key, "db1_4_frontal"].to_numpy()}
    res = Parallel(n_jobs=c5["n_jobs"])(delayed(listener_geometry)(s, idx, feat, labels, reps, membership,
                                                               c5["alignment_max_splits"], cfg.project.seed,
                                                               covariates, region_names, extra)
                                         for s, idx in groups.items())
    df = pd.DataFrame([r for rows in res for r in rows])
    out = paths.results_root
    df.to_csv(out / f"geometry_{out_tag}.csv", index=False)
    summary = {}
    for name, g in df.groupby("representation"):
        summary[name] = {"lci": summarise(g.lci.values, "greater", c5["sesoi_dz"]),
                         "alignment": summarise(g.alignment.values, "greater", c5["sesoi_dz"])}
        if "auc_mean" in g:
            summary[name]["auc_minus_half"] = summarise(g.auc_mean.values - 0.5, "greater")
    (out / f"geometry_{out_tag}_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return df, summary


def stimulus_benchmark():
    paths, cfg = load_paths(), load_analysis()
    c = cfg.section("geometry")["acoustic_covariates"]
    env = {}
    for f in cfg.dataset.processed_files.values():
        for t in load_processed(paths.processed_dir / f):
            env.setdefault(t.label, t.envelope)
    stats = {k: envelope_statistics(v, cfg.dataset.processed_sfreq, c["welch_window_s"]) for k, v in env.items()}
    meta = pd.read_csv(paths.derived_root / "network_features" / "features_eloreta_meta.csv")
    rows = []
    for s, g in meta.groupby("subject"):
        X = np.array([stats[f"{e}_{cond}."] for e, cond in zip(g.excerpt, g.condition)])
        D = crossnobis_rdm(X, g.condition.to_numpy())
        rows.append({"subject": s, "lci": lci(D), **lci_terms(D),
                     **{f"rdm_{a}_{b}": D[i, j] for i, a in enumerate(("MC", "MD", "MF", "SD", "SF"))
                        for j, b in enumerate(("MC", "MD", "MF", "SD", "SF")) if i < j}})
    df = pd.DataFrame(rows)
    out = paths.results_root / "qc"
    df.to_csv(out / "acoustic_lci.csv", index=False)
    summary = {"lci": df.lci.tolist(), "summary": summarise(df.lci.values),
               "terms": {t: float(df[t].mean()) for t in df.columns if t not in ("subject", "lci")},
               "stimulus_statistics": {k: v.round(4).tolist() for k, v in sorted(stats.items())}}
    (out / "acoustic_lci.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def aalborg_inference(tag1: str, tag2: str):
    paths, cfg = load_paths(), load_analysis()
    sesoi = cfg.section("geometry")["sesoi_dz"]
    base = paths.results_root
    a1 = pd.read_csv(base / f"geometry_{tag1}.csv")
    a2 = pd.read_csv(base / f"varx_lci_{tag2}.csv")
    gr = pd.read_csv(base / f"varx_granger_{tag2}.csv")
    prim = a1[a1.representation == "net_delta_theta"].set_index("subject")
    w2 = a2.pivot(index="subject", columns="component", values="lci")
    gm = gr.groupby(["subject", "condition"]).R.mean().unstack()
    h2b = ((gm["MD"] + gm["SD"]) - (gm["MF"] + gm["SF"])) / 2

    conf = {"H1": summarise(prim.lci.values, "greater", sesoi),
            "H1b": summarise(prim.alignment.values, "greater", sesoi),
            "H2b": summarise(h2b.values, "greater", sesoi)}
    conf["H2"] = summarise(w2["int"].values, "greater", sesoi)
    dzdiff = paired_dz_difference(w2["int"].values, w2["ext"].values, 10000, np.random.default_rng(cfg.project.seed))
    adj = holm({k: v["p_signflip"] for k, v in conf.items()})
    for k in conf:
        conf[k]["p_holm"] = adj[k]

    res = {"tests": conf,
           "representations": {r: {"lci": summarise(g.lci.values, "greater", sesoi),
                                   "alignment": summarise(g.alignment.values, "greater", sesoi)}
                               for r, g in a1.groupby("representation")},
           "acoustic_adjusted": {"lci": summarise(a1[a1.representation == "net_delta_theta_acoustic_adjusted"].lci.values,
                                                  "greater", sesoi)},
           "h2b_interaction": summarise(((gm["MD"] - gm["MF"]) - (gm["SD"] - gm["SF"])).values, "two-sided"),
           "h2_dz_int_minus_ext": dzdiff,
           "cross_decoding_auc_minus_half": summarise(prim.auc_mean.values - 0.5, "greater"),
           "varx": {f"lci_{c}": summarise(w2[c].values, "greater", sesoi) for c in ("raw", "ext", "int")}}

    tests = pd.concat([pd.DataFrame({"test": k, "subject": prim.index, "value": v})
                       for k, v in (("H1", prim.lci.values), ("H1b", prim.alignment.values))] +
                      [pd.DataFrame({"test": "int", "subject": w2.index, "value": w2["int"].values}),
                       pd.DataFrame({"test": "H2b", "subject": h2b.index, "value": h2b.values})])
    out = paths.results_root
    tests.to_csv(out / "bayes_tests.csv", index=False)
    subprocess.run([RSCRIPT, str(PROJECT_ROOT / "statistical_tests" / "one_sided_bayes_factors.R"), str(out / "bayes_tests.csv"), str(out / "bayes.json")], check=True)
    res["bayes"] = json.loads((out / "bayes.json").read_text())
    res["sensitivity"] = {"dz_80pct_alpha_.05": sesoi_resource_based(len(prim), 0.8, 0.05),
                          "dz_80pct_alpha_.0125_holm_first": sesoi_resource_based(len(prim), 0.8, 0.0125),
                          "dz_50pct_alpha_.05": sesoi_resource_based(len(prim), 0.5, 0.05)}
    (out / "inference.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
