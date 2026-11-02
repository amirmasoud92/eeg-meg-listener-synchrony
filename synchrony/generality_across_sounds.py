"""Whether a listener's synchrony generalises across sounds: Aalborg trials and CantiSong versions, pooled."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from synchrony import song_versions as c
from synchrony.correlated_components import _cov_terms, corrca_from_terms, subject_isc

def perm_rho(x, y, n=100000, seed=0):
    r = stats.spearmanr(x, y).correlation
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    g = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, g.permutation(ry))[0, 1] for _ in range(n)])
    return float(r), float((1 + (null >= r).sum()) / (1 + n))


def partial(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def split_rel(a, b):
    r = stats.spearmanr(a, b).correlation
    return float(r), float(2 * r / (1 + r))


def aalborg(paths) -> dict:
    d = pd.read_csv(paths.results_root / "aalborg_isc_speech_space.csv")
    d["excerpt"] = d.stimulus.str.split("_").str[0].astype(int)
    m = d.groupby(["subject", "condition"]).isc.mean().unstack()
    speech, inst = m[["SD", "SF"]].mean(1), m["MC"]
    mc = d[d.condition == "MC"]
    mc_odd = mc[mc.excerpt % 2 == 1].groupby("subject").isc.mean()
    mc_even = mc[mc.excerpt % 2 == 0].groupby("subject").isc.mean()
    z = np.load(paths.derived_root / "epochs" / "epoch_masks_2s.npz", allow_pickle=True)
    rej = pd.Series({int(k.split("_")[0][1:]): 1 - np.asarray(z[k], bool).mean() for k in z.files}).groupby(level=0).mean()
    out = {"n": int(len(m))}
    out["rho"], out["p"] = perm_rho(speech, inst)
    rs, rs_sb = split_rel(m.SD, m.SF)
    ri, ri_sb = split_rel(mc_odd.reindex(m.index), mc_even.reindex(m.index))
    out["reliability_speech_SD_vs_SF"], out["reliability_speech_sb"] = rs, rs_sb
    out["reliability_instrumental_odd_even"], out["reliability_instrumental_sb"] = ri, ri_sb
    out["disattenuated"] = float(out["rho"] / np.sqrt(max(rs_sb, 1e-6) * max(ri_sb, 1e-6)))
    out["partial_quality"] = partial(speech, inst, rej.reindex(m.index))
    out["also_rho_speech_song"] = float(stats.spearmanr(speech, m[["MD", "MF"]].mean(1)).correlation)
    return out


def cantisong(paths) -> tuple[dict, pd.DataFrame]:
    data = c.load_all(paths.derived_root / "cantisong", "1-4")
    songs = c.song_ids()
    terms = {}
    for (v, s), d in data.items():
        subs = c.usable(d)
        if len(subs) >= 3:
            terms[(v, s)] = _cov_terms(*c.stack(d, subs))
    rows = []
    for s in songs:
        others = [t for (v, q), t in terms.items() if q != s]
        W = corrca_from_terms(sum(t[0] for t in others), sum(t[1] for t in others), 10, 0.5, 3)[0]
        for v in ("speech", "song", "melody"):
            if (v, s) not in data:
                continue
            subs = c.usable(data[(v, s)])
            if len(subs) < 4:
                continue
            x, ok = c.stack(data[(v, s)], subs)
            for sub, val in zip(subs, subject_isc(x, W, ok).sum(1)):
                rows.append({"listener": sub, "version": v, "song": s, "isc": float(val)})
    df = pd.DataFrame(rows)
    m = df.groupby(["listener", "version"]).isc.mean().unstack()
    df["odd"] = df.song.astype(int) % 2 == 1
    half = df.groupby(["listener", "version", "odd"]).isc.mean().unstack(["version", "odd"])
    qc = pd.read_csv(paths.results_root / "C_prepare_qc.csv")
    qc["listener"] = qc.subject.str.replace("sub", "").astype(int)
    rej = qc.set_index("listener").reject_fraction.reindex(m.index)
    out = {"n": int(len(m))}
    out["rho"], out["p"] = perm_rho(m.speech, m.melody)
    rs, rs_sb = split_rel(half[("speech", True)], half[("speech", False)])
    rh, rh_sb = split_rel(half[("melody", True)], half[("melody", False)])
    out["reliability_spoken_sb"], out["reliability_hummed_sb"] = rs_sb, rh_sb
    out["disattenuated"] = float(out["rho"] / np.sqrt(max(rs_sb, 1e-6) * max(rh_sb, 1e-6)))
    out["partial_quality"] = partial(m.speech, m.melody, rej)
    out["also_rho_spoken_sung"] = float(stats.spearmanr(m.speech, m.song).correlation)
    out["also_rho_sung_hummed"] = float(stats.spearmanr(m.song, m.melody).correlation)
    return out, m
