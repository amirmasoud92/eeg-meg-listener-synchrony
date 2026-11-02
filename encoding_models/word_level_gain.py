"""Word-level gain: the true surprisal regressor against permuted versions, with the Monte Carlo rule.

Up to 500 permutations are stored one by one; the number used is the first n in 100, 200, ..., 500 at which the largest
listener's Monte Carlo standard error of the null mean is at most 10 % of the between-listener SD of the gain.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from encoding_models import gpu_encoding as WG, ridge_encoding as E, word_level_trials as P
from statistical_tests.inference import summarise

N_MAX, BLOCK, M_START, BATCH = 500, 100, 50, 12
MC_FRACTION = 0.10


def _save(f: Path, meta: dict, arrays: dict):
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, meta=json.dumps(meta, default=float), **arrays)
    os.replace(tmp, f)


def word_job(kind: str, key: str, specs: list[dict], a: int, seed: int, cache: Path, n: int) -> str:
    """CantiSong or Alice surprisal with n permutations, stored per listener."""
    f = cache / (key.replace("|", "_") + ".npz")
    if f.exists():
        return f.name
    rng = P.rng_for(seed, key)
    trials = P.canti_trials(specs, a) if kind == "canti" else P.alice_trials(specs, a)
    base = [k for k in ("acoustic", "phonetic", "word") if k in trials[0]["F"]]
    m8 = E.event_masks(trials, "words", P.WIN_S)
    folds = WG.GpuFolds(trials, base, P.N_LAGS)
    ms = {"m8": WG.MaskSet(folds, m8)}
    ctrl_s = P.perm_controls(trials, "raw_s", rng, n)
    true, ctrl = WG.run_test(folds, trials, "surprisal", ctrl_s, ms, P.N_LAGS, batch=BATCH, suff_for=("m8",))
    out = {"S_true_ch": true["m8"]["pooled"], "S_true_per": true["m8"]["per"], "S_ctrl_ch": ctrl["m8"]["pooled"],
           "S_ctrl_per": ctrl["m8"]["per"], "S_suff_true": true["m8"]["suff"], "S_suff_ctrl": ctrl["m8"]["suff"]}
    meta = {"key": key, "units": [tr.get("song", tr.get("seg")) for tr in trials], "lam": folds.lam, "n": n}
    _save(f, meta, out)
    return f.name


def mc_rule(true: np.ndarray, nulls: np.ndarray, start: int = BLOCK) -> dict:
    """true: listeners; nulls: listeners x draws (channel-mean r per control). First n (start, start + 100, ..., 500)
    at which max_i SE_i(n) <= 10 % of SD_i(delta r_i(n)); else the largest available."""
    n_avail = nulls.shape[1]
    grid = sorted({min(x, n_avail) for x in list(range(start, N_MAX + 1, BLOCK)) + [N_MAX]})
    trace = []
    for n in grid:
        se = nulls[:, :n].std(1, ddof=1) / np.sqrt(n)
        dr = true - nulls[:, :n].mean(1)
        ratio = float(se.max() / dr.std(ddof=1))
        trace.append({"n": int(n), "ratio": ratio})
        if ratio <= MC_FRACTION:
            return {"n_used": int(n), "ratio": ratio, "met": True, "trace": trace}
    return {"n_used": int(grid[-1]), "ratio": trace[-1]["ratio"], "met": False, "trace": trace}


def pick(x, alt="greater"):
    return {k: v for k, v in summarise(np.asarray(x, float), alt).items() if k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}


def _load(cache: Path) -> list[tuple[dict, dict]]:
    out = []
    for f in sorted(cache.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        with np.load(f, allow_pickle=False) as z:
            out.append((json.loads(str(z["meta"])), {k: z[k] for k in z.files if k != "meta"}))
    return out


def _test_from_channels(rows: list, tkey: str, ckey: str, chans=None) -> tuple[np.ndarray, np.ndarray]:
    ix = slice(None)
    tv, nv = [], []
    for m, z in rows:
        ix = slice(None) if chans is None else np.asarray(m["roi"])
        tv.append(float(np.mean(z[tkey][ix])))
        nv.append(z[ckey][:, ix].mean(1))
    return np.array(tv), np.array(nv)


def word_level_statistics(paths):
    """Monte Carlo rule per dataset and version, the gain per listener, and song means for the item-level contrast."""
    cache = paths.derived_root / "word_level"
    out = paths.results_root
    J = _load(cache)
    res, mc = {}, {}
    rows_l, rows_s = [], []
    groups = {}
    for m, z in J:
        ds, v, sub = m["key"].split("|")
        groups.setdefault((ds, v), []).append((m, z))
    for (ds, v), rows in sorted(groups.items()):
        alt = "two-sided" if v == "melody" else "greater"
        tv, nv = _test_from_channels(rows, "S_true_ch", "S_ctrl_ch")
        rule = mc_rule(tv, nv)
        n = rule["n_used"]
        mc[f"{ds}|{v}|pooled"] = rule
        pooled = tv - nv[:, :n].mean(1)
        mc_se = nv[:, :n].std(1, ddof=1) / np.sqrt(n)
        for i, (m, z) in enumerate(rows):
            rows_l.append({"dataset": ds, "version": v, "listener": m["key"].split("|")[2], "n_perm": n, "mc_se": mc_se[i], "pooled": float(pooled[i])})
            if ds == "canti":
                per = np.nanmean(z["S_true_per"] - z["S_ctrl_per"][:n].mean(0), axis=1)
                rows_s += [{"listener": int(m["key"].split("|")[2]), "version": v, "song": s, "dr": float(d)} for s, d in zip(m["units"], per)]
        res[f"{ds}|{v}"] = {"listeners": pick(pooled, alt)}
    L = pd.DataFrame(rows_l)
    L.to_csv(out / "word_level_listeners.csv", index=False)
    S = pd.DataFrame(rows_s)
    S.to_csv(out / "word_level_listener_song.csv", index=False)
    item = S.groupby(["song", "version"]).dr.mean().unstack()
    item.rename_axis("song_id").to_csv(out / "word_level_song_means.csv")
    res["by_songs"] = {"spoken": pick(item["speech"].dropna()), "sung": pick(item["song"].dropna()),
                       "hummed": pick(item["melody"].dropna(), "two-sided"), "sung_minus_hummed": pick((item["song"] - item["melody"]).dropna())}
    wide = L[L.dataset == "canti"].pivot(index="listener", columns="version", values="pooled")
    res["canti|sung_minus_hummed"] = pick((wide["song"] - wide["melody"]).dropna())
    res["monte_carlo"] = mc
    (out / "word_level.json").write_text(json.dumps(res, indent=2, default=float))
