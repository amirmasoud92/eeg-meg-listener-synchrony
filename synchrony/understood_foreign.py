"""Delta-band synchrony in the Aalborg and Etard data: understood against foreign speech and song.

Leave-one-stimulus-out spaces are fitted over all conditions, and ISC is computed per listener and stimulus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing import etard_recordings as nw, signal_operations as pp
from preprocessing.aalborg_trials import CleanDataset
from statistical_tests.inference import summarise
from synchrony.correlated_components import _cov_terms, corrca_from_terms, subject_isc

BANDS = {"broad": (1.0, 20.0), "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.5)}


def order(sf: float) -> int:
    m = int(np.ceil(3.3 * sf / 1.0))
    return m + (m % 2)


def band_series(x: np.ndarray, sf_in: float, step: int, lo: float, hi: float) -> np.ndarray:
    """Anti-alias at the input rate, decimate by ``step``, band-pass at the output rate (1 Hz transitions)."""
    sf = sf_in / step
    y = pp.fir_filter(x, pp.fir_kernel(min(25.0, sf / 2 - 2), sf_in, order(sf_in) // 2 * 2, False))[:, ::step]
    y = pp.fir_filter(y, pp.fir_kernel(hi, sf, order(sf), False))
    return pp.fir_filter(y, pp.fir_kernel(max(lo - 0.5, 0.25), sf, order(sf), True))


def song_data(paths, cfg, band):
    ds = CleanDataset(paths, cfg)
    a0 = cfg.section("preprocessing")["analysis_window_s"][0]
    lo, hi = band
    stim, subjects = {}, ds.subjects
    for s in subjects:
        for ct in ds.participant(s):
            t = ct.trial
            y = band_series(t.data, t.sfreq, 8, lo, hi)
            sf = t.sfreq / 8
            i0, n = int(round((5.0 - a0) * sf)), int(60 * sf)
            ok = np.repeat(ct.ok_2s[1:31], int(2 * sf))[:n]
            edge = int(2 * sf)
            ok[:edge] = ok[-edge:] = False
            stim.setdefault(t.label, {})[s] = (y[:, i0:i0 + n].astype(np.float32), ok)
    return stim, subjects


def etard_data(paths, cfg_n, band):
    base = paths.derived_root / "etard"
    ds = nw.EtardDataset(base / "preprocessed", cfg_n["participants"])
    inc = pd.read_csv(base / "epochs" / "trial_inclusion.csv").set_index("key")
    masks = dict(np.load(base / "epochs" / "epoch_masks_2s.npz"))
    subjects = sorted(inc[inc.included].subject.unique())
    lo, hi = band
    stim = {}
    for s in subjects:
        for t in ds.trials(s, with_pad=True):
            if not inc.loc[t.key].included:
                continue
            y = band_series(t.data, t.sfreq, 5, lo, hi)
            sf = t.sfreq / 5
            pad = int(round(cfg_n["pad_s"] * sf))
            y = y[:, pad:]
            ok = np.repeat(masks[t.key].astype(bool), int(2 * sf))[: y.shape[1]]
            edge = int(2 * sf)
            ok[:edge] = False
            ok[-edge:] = False
            stim.setdefault(f"{t.condition}_{t.excerpt}", {})[s] = (y[:, :ok.size].astype(np.float32), ok)
    return stim, subjects


def isc_table(stim: dict, subjects: list) -> pd.DataFrame:
    """Leave-one-stimulus-out CorrCA and subject ISC for every stimulus."""
    keys = sorted(stim)
    arrays = {}
    for k in keys:
        n = min(v[0].shape[1] for v in stim[k].values())
        D = next(iter(stim[k].values()))[0].shape[0]
        data = np.full((len(subjects), D, n), np.nan, np.float32)
        valid = np.zeros((len(subjects), n), bool)
        for i, s in enumerate(subjects):
            if s in stim[k]:
                data[i] = stim[k][s][0][:, :n]
                valid[i] = stim[k][s][1][:n]
        arrays[k] = (data, valid)
    terms = {k: _cov_terms(np.nan_to_num(d).astype(float), v) for k, (d, v) in arrays.items()}
    Rw_all = sum(t[0] for t in terms.values())
    Rb_all = sum(t[1] for t in terms.values())
    rows = []
    for k in keys:
        W, _ = corrca_from_terms(Rw_all - terms[k][0], Rb_all - terms[k][1], len(subjects))
        data, valid = arrays[k]
        isc = subject_isc(data.astype(float), W, valid)
        for i, s in enumerate(subjects):
            if s in stim[k]:
                rows.append({"subject": s, "stimulus": k, "isc": float(np.nansum(isc[i])), "isc_c1": float(isc[i, 0])})
    return pd.DataFrame(rows)


def song_contrasts(df: pd.DataFrame, labels: dict) -> list[dict]:
    df = df.assign(condition=df.stimulus.map(labels))
    w = df.groupby(["subject", "condition"]).isc.mean().unstack()
    tests = {"MD-MF": w.MD - w.MF, "SD-SF": w.SD - w.SF, "understood-not": (w.MD + w.SD - w.MF - w.SF) / 2,
             "interaction": (w.MD - w.MF) - (w.SD - w.SF), "voiced-MC": (w.MD + w.MF + w.SD + w.SF) / 4 - w.MC}
    out = [{"contrast": k, **{kk: vv for kk, vv in summarise(v.dropna().to_numpy(), "two-sided").items()
                               if kk in ("mean", "dz", "dz_ci95", "p_signflip", "n_positive", "n")}} for k, v in tests.items()]
    out.append({"contrast": "condition_means", **w.mean().round(5).to_dict()})
    return out


def etard_contrasts(df: pd.DataFrame, cfg_all) -> list[dict]:
    df = df.assign(condition=df.stimulus.str.rsplit("_", n=1).str[0])
    w = df.groupby(["subject", "condition"]).isc.mean().unstack()
    clar, comp = cfg_all.dataset["clarity_pct"], cfg_all.dataset["comprehension_pct"]
    tests = {"HR1-analogue [lb-hb]_EN - [lb-hb]_NL": (w.lb - w.hb) - (w.lbDutch - w.hbDutch),
             "EN lb-hb": w.lb - w.hb, "NL lb-hb": w.lbDutch - w.hbDutch,
             "EN-NL (cross-session)": w[["clean", "lb", "mb", "hb"]].mean(1) - w[["cleanDutch", "lbDutch", "mbDutch", "hbDutch"]].mean(1)}
    betas = []
    for s, row in w.iterrows():
        x = []
        for c, v in row.items():
            lvl, lang = (c.replace("Dutch", ""), "nl" if c.endswith("Dutch") else "en")
            x.append((clar[lvl], comp[lang][lvl], v))
        A = np.array([[1.0, a / 100, b / 100] for a, b, _ in x])
        y = np.array([v for *_, v in x])
        betas.append(np.linalg.lstsq(A, y, rcond=None)[0])
    betas = np.array(betas)
    tests["slope comprehension (controlling clarity)"] = pd.Series(betas[:, 2], index=w.index)
    tests["slope clarity (controlling comprehension)"] = pd.Series(betas[:, 1], index=w.index)
    out = [{"contrast": k, **{kk: vv for kk, vv in summarise(v.dropna().to_numpy(), "two-sided").items()
                               if kk in ("mean", "dz", "dz_ci95", "p_signflip", "n_positive", "n")}} for k, v in tests.items()]
    out.append({"contrast": "condition_means", **w.mean().round(5).to_dict()})
    return out


def arrays(paths, cfg, dataset: str, band: tuple[float, float], cfg_n: dict | None = None) -> tuple[dict, list]:
    """{stimulus: (subjects x channels x time, valid)} and the subject list, cached per dataset and band."""
    cache = paths.derived_root / "exploratory" / f"isc_arrays_{dataset}_{band[0]:g}-{band[1]:g}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        keys = z["keys"].tolist()
        return {k: (z[f"d_{i}"], z[f"v_{i}"]) for i, k in enumerate(keys)}, z["subjects"].tolist()
    stim, subjects = song_data(paths, cfg, band) if dataset == "aalborg" else etard_data(paths, cfg_n, band)
    out = {}
    for k, d in stim.items():
        n = min(v[0].shape[1] for v in d.values())
        D = next(iter(d.values()))[0].shape[0]
        data = np.zeros((len(subjects), D, n), np.float32)
        valid = np.zeros((len(subjects), n), bool)
        for i, s in enumerate(subjects):
            if s in d:
                data[i], valid[i] = d[s][0][:, :n], d[s][1][:n]
        out[k] = (data, valid)
    keys = sorted(out)
    np.savez(cache, keys=np.array(keys), subjects=np.array(subjects),
             **{f"d_{i}": out[k][0] for i, k in enumerate(keys)}, **{f"v_{i}": out[k][1] for i, k in enumerate(keys)})
    return out, subjects


def terms(arr: dict) -> dict:
    return {k: _cov_terms(d.astype(float), v) for k, (d, v) in arr.items()}


def loso_filters(arr: dict, n_sub: int, shrink: float = 0.5, ncomp: int = 3, tm: dict | None = None) -> dict:
    """CorrCA per stimulus, fitted on all other stimuli."""
    tm = tm or terms(arr)
    Rw_all, Rb_all = sum(t[0] for t in tm.values()), sum(t[1] for t in tm.values())
    return {k: corrca_from_terms(Rw_all - tm[k][0], Rb_all - tm[k][1], n_sub, shrink, ncomp)[0] for k in arr}


def song_labels(paths) -> dict:
    t = pd.read_csv(paths.processed_root / "trial_table.csv")
    return dict(zip(t.label, t.condition))


def etard_condition(stimulus: str) -> str:
    return stimulus.rsplit("_", 1)[0]
