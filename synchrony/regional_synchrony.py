"""Regional delta-band synchrony on 18 Desikan-Killiany regional time courses from the template eLORETA kernel.

Label filters are identical across listeners, so signs agree; Danish minus Finnish per region for speech and songs, with
Benjamini-Hochberg correction over the 36 tests.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from common.settings import load_analysis, load_paths
from network_states.source_model import TemplateModel, region_names
from readers.aalborg_release import load_channel_names
from statistical_tests.inference import summarise
from synchrony import understood_foreign
from synchrony.correlated_components import subject_isc

BAND = (1.0, 4.0)


def regional_contrasts():
    paths, cfg = load_paths(), load_analysis()
    c3 = cfg.section("source_model")
    out = paths.results_root
    labels = understood_foreign.song_labels(paths)
    ch = load_channel_names(paths.channel_list)
    model = TemplateModel.build(ch, cfg.dataset.preprocessed_sfreq, paths.mne_data, paths.derived_root / "source_model",
                                mindist_mm=c3["mindist_mm"])
    K = model.eloreta_kernel(c3["eloreta_lambda2"], c3["eloreta_loose"], c3["eloreta_depth"])
    names = region_names()
    Wreg = np.stack([(flip[:, None] * K[idx]).mean(0) for idx, flip in zip(model.label_vertices, model.label_flip)])

    arr, subj = understood_foreign.arrays(paths, cfg, "aalborg", BAND)
    rows = []
    for k, (d, v) in arr.items():
        reg = np.einsum("rc,nct->nrt", Wreg, d.astype(float))
        isc = subject_isc(reg, np.eye(len(names)), v)
        for i, s in enumerate(subj):
            if v[i].any():
                rows += [{"subject": s, "stimulus": k, "condition": labels[k], "region": r, "isc": float(isc[i, j])}
                         for j, r in enumerate(names)]
    df = pd.DataFrame(rows)
    df.to_csv(out / "isc_regions_aalborg.csv", index=False)
    m = df.groupby(["subject", "region", "condition"]).isc.mean().unstack()
    res = []
    for name, a, b in (("SD-SF", "SD", "SF"), ("MD-MF", "MD", "MF")):
        for r in names:
            x = (m[a] - m[b]).xs(r, level="region").to_numpy()
            s = summarise(x, "two-sided")
            res.append({"contrast": name, "region": r, "mean": s["mean"], "dz": s["dz"], "p": s["p_signflip"],
                        "n_positive": s["n_positive"], "isc_understood": float(m[a].xs(r, level="region").mean()),
                        "isc_not": float(m[b].xs(r, level="region").mean())})
    res = pd.DataFrame(res)
    res["q_bh"] = multipletests(res.p, method="fdr_bh")[1]
    res.to_csv(out / "isc_regions_contrasts.csv", index=False)

    summary = {"SD-SF": res[res.contrast == "SD-SF"].sort_values("dz", ascending=False)[["region", "dz", "p", "q_bh", "n_positive"]].round(4).to_dict("records"),
               "MD-MF": res[res.contrast == "MD-MF"].sort_values("dz", ascending=False)[["region", "dz", "p", "q_bh", "n_positive"]].round(4).to_dict("records"),
               "n_significant_q05": res.groupby("contrast").apply(lambda g: int((g.q_bh < 0.05).sum()), include_groups=False).to_dict(),
               "negative_regional_isc_fraction": float((df.isc < 0).mean())}
    (out / "isc_regions.json").write_text(json.dumps(summary, indent=2, default=str))
