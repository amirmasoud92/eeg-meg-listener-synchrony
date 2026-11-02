"""The specification search on the Aalborg and Etard data.

Every combination of feature family, band set and granularity in every source space, with uncorrected and search-wide
p-values; selection by the hypothesis-neutral mean distance, and transfer of a selected specification to the other dataset.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common.settings import load_analysis, load_paths, rng
from network_states.network_features import network_membership
from network_states.representational_geometry import CONDITIONS, comprehension_alignment, cross_decoding_auc, crossnobis_pair, crossnobis_rdm, lci
from network_states.specification_search import Spec, build, signflip_maxt, specifications
from readers import etard_release as etard

SONG_STATS = ("lci", "alignment", "auc", "neutral")
ETARD_STATS = ("delta", "d_en", "d_nl", "neutral")


def song_eval(feat, meta, labels, spec: Spec, membership, seed, space):
    X_all = build(feat, spec, membership)
    rows = []
    for s, idx in meta.groupby("subject").indices.items():
        X, lab = X_all[idx], labels[idx]
        D = crossnobis_rdm(X, lab)
        iu = np.triu_indices(len(CONDITIONS), 1)
        rows.append({"space": space, "spec": spec.name, "subject": s, "lci": lci(D), "neutral": float(D[iu].mean()),
                     "alignment": comprehension_alignment(X, lab, max_splits=50, rng=rng(seed, "x1", space, spec.name, s)),
                     "auc": cross_decoding_auc(X, lab)["mean"] - 0.5, "n_features": X.shape[1]})
    return rows


def etard_eval(feat, meta, labels, spec: Spec, membership, cfg, space):
    X_all = build(feat, spec, membership)
    rows = []
    for s, idx in meta.groupby("subject").indices.items():
        X, lab = X_all[idx], labels[idx]
        ce, cn = cfg["contrast"]["EN"], cfg["contrast"]["NL"]
        d_en, d_nl = crossnobis_pair(X, lab, *ce), crossnobis_pair(X, lab, *cn)
        within = [crossnobis_pair(X, lab, a, b) for conds in cfg["sessions"].values()
                  for a, b in itertools.combinations(conds, 2)]
        rows.append({"space": space, "spec": spec.name, "subject": s, "d_en": d_en, "d_nl": d_nl, "delta": d_en - d_nl,
                     "neutral": float(np.mean(within)), "n_features": X.shape[1]})
    return rows


def inference(df: pd.DataFrame, stats: tuple[str, ...]) -> pd.DataFrame:
    out = []
    keys = df[["space", "spec"]].drop_duplicates().apply(tuple, axis=1).tolist()
    for stat in stats:
        wide = df.pivot_table(index="subject", columns=["space", "spec"], values=stat)[keys]
        t, p_unc, p_fwe = signflip_maxt(wide.to_numpy(), alternative="greater")
        for (space, spec), ti, pu, pf in zip(keys, t, p_unc, p_fwe):
            col = wide[(space, spec)]
            out.append({"stat": stat, "space": space, "spec": spec, "mean": float(col.mean()),
                        "dz": float(col.mean() / col.std(ddof=1)), "t": float(ti), "p_unc": float(pu), "p_fwe": float(pf)})
    res = pd.DataFrame(out)
    parts = res.spec.str.split("|", expand=True)
    res["granularity"], res["family"], res["bands"] = parts[0], parts[1], parts[2]
    return res


def search_both_datasets(n_jobs: int):
    paths, cfg_a = load_paths(), load_analysis()
    cfg_e = etard.load_config().section("network")
    out = paths.results_root
    specs = specifications()
    seed = cfg_a.project.seed

    rows = []
    for space in ("eloreta", "lcmv", "sensor"):
        arr = dict(np.load(paths.derived_root / "network_features" / f"features_{space}.npz", allow_pickle=True))
        meta = pd.read_csv(paths.derived_root / "network_features" / f"features_{space}_meta.csv")
        feat = {k: v for k, v in arr.items() if k != "keys"}
        labels = meta.condition.to_numpy()
        regions = np.load(paths.derived_root / "source_model" / "filters_S1.npz")["regions"].tolist()
        mem = network_membership(regions)
        res = Parallel(n_jobs=n_jobs)(delayed(song_eval)(feat, meta, labels, sp, mem, seed, space) for sp in specs)
        rows += [r for rr in res for r in rr]
    song = pd.DataFrame(rows)
    song.to_csv(out / "multiverse_aalborg_participants.csv", index=False)

    rows = []
    base = paths.derived_root / "etard"
    for space in ("eloreta", "sensor"):
        arr = dict(np.load(base / "features" / f"features_{space}.npz", allow_pickle=True))
        meta = pd.read_csv(base / "features" / f"features_{space}_meta.csv")
        feat = {k: v for k, v in arr.items() if k != "keys"}
        labels = meta.condition.to_numpy()
        regions = np.load(next((base / "filters").glob("filters_*.npz")))["regions"].tolist()
        mem = network_membership(regions)
        res = Parallel(n_jobs=n_jobs)(delayed(etard_eval)(feat, meta, labels, sp, mem, cfg_e, space) for sp in specs)
        rows += [r for rr in res for r in rr]
    eta = pd.DataFrame(rows)
    eta.to_csv(out / "multiverse_etard_participants.csv", index=False)

    inf_song = inference(song, SONG_STATS)
    inf_eta = inference(eta, ETARD_STATS)
    inf_song.to_csv(out / "multiverse_aalborg_inference.csv", index=False)
    inf_eta.to_csv(out / "multiverse_etard_inference.csv", index=False)

    def best(inf, stat):
        r = inf[inf.stat == stat].sort_values("t", ascending=False).iloc[0]
        return r.space, r.spec

    def lookup(inf, stat, space, spec):
        r = inf[(inf.stat == stat) & (inf.space == space) & (inf.spec == spec)].iloc[0]
        return {k: (float(r[k]) if k != "spec" else r[k]) for k in ("mean", "dz", "p_unc", "p_fwe")}

    map_space = lambda s: "eloreta" if s == "lcmv" else s
    sel_song = best(inf_song, "neutral")
    sel_eta = best(inf_eta, "neutral")
    summary = {
        "n_specs_per_space": len(specs),
        "aalborg": {s: {"n_p_unc_05": int((g.p_unc < 0.05).sum()), "n_p_fwe_05": int((g.p_fwe < 0.05).sum()),
                        "best": g.sort_values("t", ascending=False).iloc[0][["space", "spec", "dz", "p_unc", "p_fwe"]].to_dict(),
                        "median_dz": float(g.dz.median())}
                    for s, g in inf_song.groupby("stat")},
        "etard": {s: {"n_p_unc_05": int((g.p_unc < 0.05).sum()), "n_p_fwe_05": int((g.p_fwe < 0.05).sum()),
                      "best": g.sort_values("t", ascending=False).iloc[0][["space", "spec", "dz", "p_unc", "p_fwe"]].to_dict(),
                      "median_dz": float(g.dz.median())}
                  for s, g in inf_eta.groupby("stat")},
        "neutral_selection": {
            "aalborg_selected": sel_song, "aalborg_selected_tests": {s: lookup(inf_song, s, *sel_song) for s in ("lci", "alignment", "auc")},
            "etard_selected": sel_eta, "etard_selected_tests": {s: lookup(inf_eta, s, *sel_eta) for s in ("delta", "d_en", "d_nl")},
        },
        "transfer": {
            "aalborg_neutral_to_etard_delta": lookup(inf_eta, "delta", map_space(sel_song[0]), sel_song[1]),
            "etard_neutral_to_aalborg": {s: lookup(inf_song, s, sel_eta[0], sel_eta[1]) for s in ("lci", "alignment", "auc")},
            "aalborg_best_alignment_to_etard_delta": lookup(inf_eta, "delta", map_space(best(inf_song, "alignment")[0]), best(inf_song, "alignment")[1]),
            "aalborg_best_lci_to_etard_delta": lookup(inf_eta, "delta", map_space(best(inf_song, "lci")[0]), best(inf_song, "lci")[1]),
            "aalborg_best_auc_to_etard_delta": lookup(inf_eta, "delta", map_space(best(inf_song, "auc")[0]), best(inf_song, "auc")[1]),
            "etard_best_delta_to_aalborg": {s: lookup(inf_song, s, *best(inf_eta, "delta")) for s in ("lci", "alignment", "auc")},
        },
    }
    summary["transfer"]["specs"] = {"aalborg_best_alignment": best(inf_song, "alignment"), "aalborg_best_lci": best(inf_song, "lci"),
                                    "aalborg_best_auc": best(inf_song, "auc"), "etard_best_delta": best(inf_eta, "delta")}
    (out / "multiverse_summary.json").write_text(json.dumps(summary, indent=2, default=str))
