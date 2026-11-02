"""VARX tests on the Aalborg data and the selection of the recurrent order."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common.settings import load_analysis, load_paths, rng
from network_states.network_features import BAND_NAMES, bands_from_config, network_membership
from network_states.varx_analysis import COMPONENTS, VarxSettings, crossfit_geometry, granger_and_fits, lci_rows
from network_states.varx_order import select_order
from network_states.varx_trials import VarxTrial, nan_bad_epochs, prepare_trial
from preprocessing.aalborg_trials import CleanDataset
from readers.aalborg_release import load_processed
from statistical_tests.inference import summarise

def settings(cfg, na: int) -> VarxSettings:
    c6, c2, c4 = cfg.section("varx"), cfg.section("preprocessing"), cfg.section("network_features")
    return VarxSettings(na=na, nb=int(round(c6["input_lag_max_ms"] / 1e3 * c6["sfreq"])) + 1, lam=c6["lambda"],
                        sfreq=c6["sfreq"], epoch_s=c2["connectivity_epoch_s"], spectral_epoch_s=c2["spectral_epoch_s"],
                        n_boot=c6["n_bootstraps"], n_splits=c6["n_crossfit_splits"],
                        bands=tuple(bands_from_config(c4["bands"])),
                        band_idx=tuple(BAND_NAMES.index(b) for b in c6["h2_representation"]["bands"]))


def listener_varx(s, paths, cfg, label_of, na, seed, replicate=0):
    c6 = cfg.section("varx")
    d = np.load(paths.derived_root / "varx" / f"varx_inputs_S{s}.npz")
    trials = [VarxTrial(k, s, label_of[k], y, x, ok) for k, y, x, ok in zip(d["keys"].tolist(), d["y"], d["x"], d["ok"])]
    regions = np.load(paths.derived_root / "source_model" / f"filters_S{s}.npz")["regions"].tolist()
    ridx = {r: i for i, r in enumerate(regions)}
    st = settings(cfg, na)
    g = rng(seed, "aim2", s, replicate)
    gr, fits = granger_and_fits(trials, st, [(ridx[a], ridx[b]) for a, b in c6["h2b_edges"]], g)
    rdms = crossfit_geometry(trials, st, network_membership(regions), seed, g)
    tag = {"subject": s}
    return [{**tag, **r} for r in lci_rows(rdms)], [{**tag, **r} for r in gr], [{**tag, **r} for r in fits]


def varx_tests(na, tag, n_jobs):
    paths, cfg = load_paths(), load_analysis()
    c6 = cfg.section("varx")
    meta = pd.read_csv(paths.derived_root / "network_features" / "features_eloreta_meta.csv")
    label_of = dict(zip(meta.key, meta.condition))
    res = Parallel(n_jobs=n_jobs)(delayed(listener_varx)(s, paths, cfg, label_of, na, cfg.project.seed)
                                  for s in sorted(meta.subject.unique()))
    lci_df, gr, fits = (pd.DataFrame([r for x in res for r in x[i]]) for i in range(3))
    out = paths.results_root
    for df, nm in ((lci_df, "lci"), (gr, "granger"), (fits, "fits")):
        df.to_csv(out / f"varx_{nm}_{tag}.csv", index=False)
    wide = lci_df.pivot(index="subject", columns="component", values="lci")
    gm = gr.groupby(["subject", "condition"]).R.mean().unstack()
    sesoi = cfg.section("geometry")["sesoi_dz"]
    summary = {f"lci_{c}": summarise(wide[c].values, "greater", sesoi) for c in COMPONENTS}
    summary["lci_int_minus_ext"] = summarise((wide["int"] - wide["ext"]).values, "greater")
    summary["h2b_comprehension_effect_R"] = summarise((((gm["MD"] + gm["SD"]) - (gm["MF"] + gm["SF"])) / 2).values,
                                                      "greater")
    summary["h2b_interaction_R"] = summarise(((gm["MD"] - gm["MF"]) - (gm["SD"] - gm["SF"])).values, "two-sided")
    summary["model"] = {"spectral_radius_max": float(fits.spectral_radius.max()),
                        "explained_by_input_median": float(fits.explained_by_input.median()),
                        "T_min": int(fits["T"].min()), "na": na}
    (out / f"varx_{tag}_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def prepare_listener(s, paths, cfg, envelopes):
    c6, c2 = cfg.section("varx"), cfg.section("preprocessing")
    ds = CleanDataset(paths, cfg)
    W = np.load(paths.derived_root / "source_model" / f"filters_S{s}.npz")["eloreta"]
    keys, conds, Ys, Xs, oks = [], [], [], [], []
    for ct in ds.participant(s, pad_s=c6["prep_pad_s"]):
        t = ct.trial
        y, x, ok = prepare_trial(W @ t.data, ct.pad / t.sfreq, envelopes[(s, t.label)], ct.ok_2s,
                                 c2["analysis_window_s"][0], tuple(c6["window_s"]), t.sfreq, c6["sfreq"],
                                 cfg.dataset.processed_sfreq, c6["envelope_pad_s"], c2["connectivity_epoch_s"])
        keys.append(t.key), conds.append(t.condition), Ys.append(y), Xs.append(x), oks.append(ok)
    out = paths.derived_root / "varx"
    np.savez(out / f"varx_inputs_S{s}.npz", keys=np.array(keys), conditions=np.array(conds), y=np.stack(Ys),
             x=np.stack(Xs), ok=np.stack(oks))
    epoch_n = int(round(c2["connectivity_epoch_s"] * c6["sfreq"]))
    Yn = [nan_bad_epochs(y, ok, epoch_n) for y, ok in zip(Ys, oks)]
    nb = int(round(c6["input_lag_max_ms"] / 1e3 * c6["sfreq"])) + 1
    # BIC's optimum grows with sample size, so order is selected at the size of one condition's data:
    # random groups of ``order_group_size`` trials.
    rng = np.random.default_rng([cfg.project.seed, s])
    idx = rng.permutation(len(Yn))
    size = c6["order_group_size"]
    groups = [idx[i:i + size] for i in range(0, len(idx) - size + 1, size)]
    tabs = []
    for gi, g in enumerate(groups):
        tab = select_order([Yn[i] for i in g], [Xs[i] for i in g], list(range(1, c6["na_max"] + 1)), nb,
                           c6["lambda"], c6["order_maxlag"])
        tab.insert(0, "group", gi)
        tabs.append(tab)
    tab = pd.concat(tabs, ignore_index=True).groupby("na", as_index=False).agg(
        bic=("bic", "mean"), aic=("aic", "mean"), cv_nmse=("cv_nmse", "mean"), cv_nmse_sem=("cv_nmse_sem", "mean"))
    tab.insert(0, "subject", s)
    return tab


def select_recurrent_order():
    paths, cfg = load_paths(), load_analysis()
    c6 = cfg.section("varx")
    out = paths.derived_root / "varx"
    out_q = paths.results_root / "qc"
    envelopes = {}
    for f in cfg.dataset.processed_files.values():
        for t in load_processed(paths.processed_dir / f):
            envelopes[(t.participant, t.label)] = t.envelope
    subjects = CleanDataset(paths, cfg).subjects
    tabs = Parallel(n_jobs=6)(delayed(prepare_listener)(s, paths, cfg, {k: v for k, v in envelopes.items() if k[0] == s})
                              for s in subjects)
    order = pd.concat(tabs, ignore_index=True)
    order.to_csv(out_q / "order_selection.csv", index=False)
    best = order.loc[order.groupby("subject").bic.idxmin(), ["subject", "na"]].rename(columns={"na": "na_bic"})
    best_cv = order.loc[order.groupby("subject").cv_nmse.idxmin(), ["subject", "na"]].rename(columns={"na": "na_cv"})
    best = best.merge(best_cv, on="subject")
    na_global = int(np.round(np.median(best.na_bic)))
    cv_at = order[order.na == na_global].set_index("subject")
    cv_min = order.loc[order.groupby("subject").cv_nmse.idxmin()].set_index("subject")
    within = ((cv_at.cv_nmse - cv_min.cv_nmse) <= cv_min.cv_nmse_sem).mean()
    summary = {"na_bic_per_subject": best.set_index("subject").na_bic.to_dict(),
               "na_cv_per_subject": best.set_index("subject").na_cv.to_dict(),
               "na_selected": na_global,
               "fraction_subjects_cv_within_1sem_of_min_at_selected": float(within),
               "cv_nmse_at_selected_median": float(cv_at.cv_nmse.median())}
    (out_q / "order_summary.json").write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
