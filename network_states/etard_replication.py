"""The network-state analysis on the Etard data: rejection, source filters, features, the within-session contrast and its stimulus benchmark."""
from __future__ import annotations

import itertools
import json
from fractions import Fraction

import numpy as np
import pandas as pd
import scipy.signal as ss
from joblib import Parallel, delayed

from common.settings import rng
from encoding_models import speech_envelope as tk
from network_states.envelope_statistics import AcousticCovariates, envelope_statistics
from network_states.network_features import BAND_NAMES, bands_from_config, extract, network_membership, stacked_vector
from network_states.representational_geometry import crossnobis_pair
from network_states.source_model import TemplateModel, region_names
from network_states.spectral_parameters import SpecparamSettings
from preprocessing import etard_recordings as nw
from preprocessing.epoch_rejection import reject
from readers import etard_release as etard
from statistical_tests.inference import jzs_bf10, summarise

FIELDS = ("dwpli", "aec", "power", "exponent", "offset", "r_squared", "fit_error")


def dirs(paths):
    base = paths.derived_root / "etard"
    d = {"pre": base / "preprocessed", "epochs": base / "epochs", "filters": base / "filters", "features": base / "features",
         "qc": paths.results_root / "qc" / "etard"}
    return d


def preprocess_sessions(cfg_all, cfg, c2, d, n_jobs):
    root = cfg_all.root
    jobs = [(sid, sess, conds) for sid in cfg["participants"] for sess, conds in cfg["sessions"].items()]
    metas = Parallel(n_jobs=n_jobs)(delayed(nw.preprocess_session)(
        root, sid, sess, conds, cfg, c2["connectivity_epoch_s"], d["pre"], int(rng(cfg["seed"], "ica", sid, sess).integers(2**31)))
        for sid, sess, conds in jobs)
    rows = [{"subject": m["subject"], "session": m["session"], "n_bads": len(m["bads"]), "bads": ",".join(m["bads"]),
             "rank": m["rank"], "n_ica_removed": len(m["ica_excluded"]), "ica_max_score": m["ica_max_score"]}
            for m in metas]
    pd.DataFrame(rows).to_csv(d["qc"] / "preprocessing.csv", index=False)


def _reject_participant(ds, sid, c2, seed):
    rows, masks = [], {}
    avail = ds.available(sid)
    for sess in ("EN", "NL"):
        trials = ds.trials(sid, sess)
        res = reject(trials, ds.channels, c2["connectivity_epoch_s"], seed=seed,
                     max_bad_fraction=c2["max_bad_epoch_fraction"], method=c2["epoch_rejection"],
                     z_threshold=c2["faster_z"], channel_robust_z=c2["channel_robust_z"])
        for t in trials:
            ok_rej = res.epoch_ok[t.key]
            av = avail[t.key][: ok_rej.size]
            ok = ok_rej & av
            masks[t.key] = ok
            bad_frac = float((~ok_rej[av]).mean()) if av.any() else 1.0
            rows.append({"subject": sid, "session": sess, "condition": t.condition, "part": t.excerpt, "key": t.key,
                         "order": t.presentation_index, "n_epochs_2s": ok.size, "n_available_2s": int(av.sum()),
                         "n_clean_2s": int(ok.sum()), "bad_fraction": bad_frac,
                         "trial_retained": bad_frac <= c2["max_bad_epoch_fraction"]})
    return rows, masks


def reject_epochs(cfg, c2, d, n_jobs, seed):
    ds = nw.EtardDataset(d["pre"], cfg["participants"])
    res = Parallel(n_jobs=n_jobs)(delayed(_reject_participant)(ds, sid, c2, seed) for sid in cfg["participants"])
    table = pd.DataFrame([r for rows, _ in res for r in rows])
    masks = {k: v for _, m in res for k, v in m.items()}
    kept = table[table.trial_retained]
    conds = [c for cs in cfg["sessions"].values() for c in cs]
    counts = kept.pivot_table(index="subject", columns="condition", values="key", aggfunc="count").reindex(
        index=cfg["participants"], columns=conds).fillna(0).astype(int)
    ok = (counts >= cfg["min_parts_per_condition"]).all(axis=1)
    table["participant_retained"] = table.subject.map(ok)
    table["included"] = table.trial_retained & table.participant_retained
    table["equalised_n_2s"] = table.subject.map(table[table.included].groupby("subject").n_clean_2s.min())
    table.to_csv(d["epochs"] / "trial_inclusion.csv", index=False)
    np.savez_compressed(d["epochs"] / "epoch_masks_2s.npz", **masks)
    inc = table[table.included]
    summary = {"participants_excluded": ok.index[~ok].tolist(),
               "parts_excluded": table.loc[~table.trial_retained, "key"].tolist(),
               "epochs_rejected_fraction_by_session": {s: float(1 - g.n_clean_2s.sum() / g.n_available_2s.sum())
                                                       for s, g in table.groupby("session")},
               "english_masked_fraction_NL": float(1 - table[table.session == "NL"].n_available_2s.sum()
                                                   / table[table.session == "NL"].n_epochs_2s.sum()),
               "equalised_n_2s": inc.groupby("subject").equalised_n_2s.first().to_dict()}
    (d["qc"] / "rejection_summary.json").write_text(json.dumps(summary, indent=2, default=str))


def sensor_selection(model: TemplateModel, ch_names: list[str]) -> np.ndarray:
    """One channel per region: the channel with the largest leadfield for the label in the region's
    hemisphere, assigned greedily from the strongest label (the Aalborg sensor-space mapping)."""
    names = region_names()
    G = model.label_leadfields() / np.array([len(v) for v in model.label_vertices])
    pos = model.info.get_montage().get_positions()["ch_pos"]
    x = np.array([pos[c][0] for c in ch_names])
    used, W = set(), np.zeros((len(names), len(ch_names)))
    for r in np.argsort(-np.abs(G).max(0)):
        left = names[r].endswith("_lh")
        cand = [i for i in np.argsort(-np.abs(G[:, r])) if ((x[i] < 0) if left else (x[i] > 0)) and i not in used]
        W[r, cand[0]] = 1.0
        used.add(cand[0])
    return W


def usable(inc: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    """Included parts, plus the retained parts of explicitly listed excluded participants (robustness)."""
    return inc[inc.included | (inc.subject.isin(extra) & inc.trial_retained)]


def source_filters(cfg, cfg_a, d, paths, extra=()):
    c3 = cfg_a.section("source_model")
    ds = nw.EtardDataset(d["pre"], cfg["participants"])
    inc = pd.read_csv(d["epochs"] / "trial_inclusion.csv")
    masks = dict(np.load(d["epochs"] / "epoch_masks_2s.npz"))
    ch = ds.channels
    sf = float(next(iter(ds.meta.values()))["sfreq"])
    model = TemplateModel.build(ch, sf, paths.mne_data, d["filters"], mindist_mm=c3["mindist_mm"])
    K = model.eloreta_kernel(c3["eloreta_lambda2"], c3["eloreta_loose"], c3["eloreta_depth"])
    W_s = sensor_selection(model, ch)
    rows = []
    use = usable(inc, list(extra))
    for sid in sorted(use.subject.unique()):
        if extra and sid not in extra:
            continue
        keys = set(use[use.subject == sid].key)
        cov, n = 0.0, 0
        for t in ds.trials(sid):
            if t.key not in keys:
                continue
            ok = masks[t.key]
            nep = int(round(2.0 * t.sfreq))
            m = np.zeros(t.data.shape[1], bool)
            m[: ok.size * nep] = np.repeat(ok, nep)
            x = t.data[:, m]
            x = x - x.mean(axis=1, keepdims=True)
            cov = cov + x @ x.T
            n += x.shape[1]
        cov = cov / (n - 1)
        ev = np.linalg.eigvalsh(cov)[::-1]
        rank = int((ev / ev[0] > c3["rank_tol"]).sum())
        W_e = model.regional_filters(K, cov)
        np.savez(d["filters"] / f"filters_{sid}.npz", eloreta=W_e, sensor=W_s, regions=np.array(region_names()), rank=rank)
        rows.append({"subject": sid, "rank": rank, "clean_samples": n})
    pd.DataFrame(rows).to_csv(d["qc"] / f"source_diagnostics{'_extra' if extra else ''}.csv", index=False)
    chosen = {region_names()[r]: ch[int(np.argmax(W_s[r]))] for r in range(W_s.shape[0])}
    (d["qc"] / "sensor_selection.json").write_text(json.dumps(chosen, indent=2))


def _features_participant(sid, d, cfg, cfg_a, source, extra=()):
    c2, c4 = cfg_a.section("preprocessing"), cfg_a.section("network_features")
    ds = nw.EtardDataset(d["pre"], cfg["participants"])
    inc = pd.read_csv(d["epochs"] / "trial_inclusion.csv").set_index("key")
    masks = dict(np.load(d["epochs"] / "epoch_masks_2s.npz"))
    W = np.load(d["filters"] / f"filters_{sid}.npz")[source]
    bands = bands_from_config(c4["bands"])
    sbands = bands_from_config(c4["spectral_bands"])
    spec = SpecparamSettings(**{k: tuple(v) if isinstance(v, list) else v for k, v in c4["specparam"].items()})
    rows, feats = [], []
    use = usable(inc.reset_index(), list(extra)).set_index("key")
    n_eq_extra = int(use[use.subject == sid].n_clean_2s.min()) if sid in extra else None
    for t in ds.trials(sid, with_pad=True):
        if t.key not in use.index:
            continue
        r = use.loc[t.key]
        n_eq = n_eq_extra if n_eq_extra is not None else int(r.equalised_n_2s)
        pad = int(round(cfg["pad_s"] * t.sfreq))
        f = extract(W @ t.data, pad, masks[t.key].astype(bool), n_eq, t.sfreq, bands, sbands, spec,
                    c2["connectivity_epoch_s"], c2["spectral_epoch_s"], c4["n_bootstraps"],
                    rng(cfg["seed"], "equalise", t.key), key=t.key)
        feats.append(f)
        rows.append({"key": t.key, "subject": sid, "session": r.session, "condition": t.condition, "part": t.excerpt,
                     "order": int(r.order), "n_clean_2s": int(r.n_clean_2s), "n_equalised_2s": n_eq})
    return rows, feats


def session_features(cfg, cfg_a, d, n_jobs, source, extra=()):
    inc = pd.read_csv(d["epochs"] / "trial_inclusion.csv")
    subjects = sorted(usable(inc, list(extra)).subject.unique())
    res = Parallel(n_jobs=n_jobs)(delayed(_features_participant)(s, d, cfg, cfg_a, source, tuple(extra)) for s in subjects)
    source_tag = source + ("_extra" if extra else "")
    meta = pd.DataFrame([r for rows, _ in res for r in rows])
    feats = [f for _, fs in res for f in fs]
    arrays = {k: np.stack([getattr(f, k) for f in feats]) for k in FIELDS}
    np.savez_compressed(d["features"] / f"features_{source_tag}.npz", keys=meta.key.values, **arrays)
    meta.to_csv(d["features"] / f"features_{source_tag}_meta.csv", index=False)
    c4 = cfg_a.section("network_features")
    summary = {"n_parts": len(meta), "n_subjects": int(meta.subject.nunique()),
               "any_nan": {k: bool(np.isnan(v).any()) for k, v in arrays.items()},
               "specparam_r2_median": float(np.median(arrays["r_squared"])),
               "specparam_r2_below_min": float((arrays["r_squared"] < c4["min_specparam_r2"]).mean()),
               "dwpli_mean_by_band": dict(zip(BAND_NAMES, arrays["dwpli"].mean(axis=(0, 2)).round(4).tolist())),
               "aec_mean_by_band": dict(zip(BAND_NAMES, arrays["aec"].mean(axis=(0, 2)).round(4).tolist())),
               "by_session_dwpli_theta": meta.assign(v=arrays["dwpli"][:, 1].mean(1)).groupby("session").v.mean().round(4).to_dict()}
    (d["qc"] / f"features_summary_{source_tag}.json").write_text(json.dumps(summary, indent=2))


def load(paths, source: str):
    base = paths.derived_root / "etard"
    arr = dict(np.load(base / "features" / f"features_{source}.npz", allow_pickle=True))
    meta = pd.read_csv(base / "features" / f"features_{source}_meta.csv")
    assert (arr["keys"] == meta.key.values).all()
    regions = np.load(next((base / "filters").glob("filters_*.npz")))["regions"].tolist()
    return arr, meta, network_membership(regions)


def patterns(arr: dict, rep: dict, membership: np.ndarray) -> np.ndarray:
    bands = [BAND_NAMES.index(b) for b in rep["bands"]]
    return stacked_vector({k: v for k, v in arr.items() if k != "keys"}, bands, rep["summarised"], membership)


def _pair(X: np.ndarray, lab: np.ndarray, a: str, b: str, covariates=None) -> float:
    """Crossnobis distance, or NaN when a condition has fewer than two parts (excluded parts)."""
    if min((lab == a).sum(), (lab == b).sum()) < 2:
        return float("nan")
    return crossnobis_pair(X, lab, a, b, covariates)


def participant_contrasts(X: np.ndarray, lab: np.ndarray, cfg: dict, covariates: np.ndarray | None = None) -> dict:
    ce, cn = cfg["contrast"]["EN"], cfg["contrast"]["NL"]
    d_en = crossnobis_pair(X, lab, ce[0], ce[1], covariates)
    d_nl = crossnobis_pair(X, lab, cn[0], cn[1], covariates)
    out = {"d_en": d_en, "d_nl": d_nl, "delta": d_en - d_nl, "delta_scaled": d_en - cfg["sensitivity_ratio"] * d_nl}
    if covariates is None:
        en, nl = cfg["sessions"]["EN"], cfg["sessions"]["NL"]
        comp = cfg["graded_comprehension_change"]["EN"]
        num, den = 0.0, 0.0
        for (i, a), (j, b) in itertools.combinations(enumerate(en), 2):
            w = abs(comp[a] - comp[b])
            num += w * (_pair(X, lab, a, b) - _pair(X, lab, nl[i], nl[j]))
            den += w
        out["delta_graded"] = num / den
        for a, b in zip(en, nl):
            out[f"cross_{a}"] = _pair(X, lab, a, b)
    return out


def rep_spec(cfg_a, name: str) -> dict:
    return next(r for r in cfg_a.section("geometry")["representations"] if r["name"] == name)


def run_delta(paths, cfg, cfg_a, source: str, covariates=None):
    arr, meta, membership = load(paths, source)
    lab_all = meta.condition.to_numpy()
    X_all = patterns(arr, rep_spec(cfg_a, cfg["representation"]), membership)
    rows = []
    for sid, idx in meta.groupby("subject").indices.items():
        cov = None if covariates is None else covariates[idx]
        r = participant_contrasts(X_all[idx], lab_all[idx], cfg, cov)
        rows.append({"subject": sid, **r})
    df = pd.DataFrame(rows)
    res = {"primary": summarise(df.delta.values, "greater"),
           "bf10_plus": jzs_bf10(df.delta.values),
           "d_en": summarise(df.d_en.values, "greater"), "d_nl": summarise(df.d_nl.values, "greater"),
           "sensitivity_scaled": summarise(df.delta_scaled.values, "greater")}
    if "delta_graded" in df:
        res["graded"] = summarise(df.delta_graded.dropna().values, "greater")
        res["cross_session"] = {c: summarise(df[c].dropna().values, "greater") for c in df.columns if c.startswith("cross_")}
    return df, res


def mixture_envelopes(cfg_all, cfg, sfreq_out: int) -> dict[tuple[str, int], np.ndarray]:
    """Compressed broadband envelope of the presented mixture per part, at ``sfreq_out`` Hz from onset."""
    root, snr = cfg_all.root, cfg_all.dataset["snr_db"]
    env_cfg = cfg_all.section("envelope")
    out = {}
    for sess, conds in cfg["sessions"].items():
        for cond in conds:
            level = snr.get(cond.replace("Dutch", ""))
            for k, f in enumerate(etard.stimulus_files(root, cond), start=1):
                st = etard.load_stimulus(f, cond, k)
                e1k = tk.broadband_envelope(st.mixture(level), st.fs, env_cfg)
                fr = Fraction(sfreq_out, 1000)
                out[(cond, k)] = ss.resample_poly(e1k, fr.numerator, fr.denominator)
    return out


def epochs_envelope(env: np.ndarray, sf: int, ok: np.ndarray, start_s: float, epoch_s: float) -> np.ndarray:
    n = int(round(epoch_s * sf))
    s0 = int(round(start_s * sf))
    segs = [env[s0 + i * n: s0 + (i + 1) * n] for i in np.flatnonzero(ok)]
    return np.concatenate([s for s in segs if s.size == n])


def run_acoustic(paths, cfg_all, cfg, cfg_a, source: str):
    """Stimulus-only benchmark and trial-level acoustic covariates (true stimulus identities)."""
    sf = int(cfg["acoustic_envelope_sfreq"])
    welch = cfg_a.section("geometry")["acoustic_covariates"]["welch_window_s"]
    env = mixture_envelopes(cfg_all, cfg, sf)
    masks = dict(np.load(paths.derived_root / "etard" / "epochs" / "epoch_masks_2s.npz"))
    _, meta, _ = load(paths, source)
    stats, per_key_env = [], {}
    for r in meta.itertuples():
        e = epochs_envelope(env[(r.condition, r.part)], sf, masks[r.key].astype(bool), cfg["window_start_s"], 2.0)
        per_key_env[r.key] = e
        stats.append(envelope_statistics(e, sf, welch))
    S = np.array(stats)
    lab = meta.condition.to_numpy()
    rows = []
    for sid, idx in meta.groupby("subject").indices.items():
        rows.append({"subject": sid, **participant_contrasts(S[idx], lab[idx], cfg)})
    df = pd.DataFrame(rows)
    ac = AcousticCovariates(per_key_env, sf, cfg_a.section("geometry")["acoustic_covariates"]["pca_variance"], welch)
    covs = ac(meta.key.tolist())
    res = {"delta": summarise(df.delta.values, "greater"), "d_en": float(df.d_en.mean()), "d_nl": float(df.d_nl.mean()),
           "delta_mean": float(df.delta.mean()), "delta_relative": float(df.delta.mean() / df[["d_en", "d_nl"]].mean().mean()),
           "graded": float(df.delta_graded.mean()), "n_covariate_components": int(ac.k),
           "explained": ac.explained.round(3).tolist(),
           "condition_statistics": pd.DataFrame(S, columns=["m0.5-1", "m1-2", "m2-4", "m4-8", "log_fpeak", "log_sd"])
           .assign(condition=lab).groupby("condition").mean().round(4).to_dict("index")}
    return df, res, covs
