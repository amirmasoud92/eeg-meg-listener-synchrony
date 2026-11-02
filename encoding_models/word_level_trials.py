"""Trials, regressors and within-unit permutation of surprisal for the word-level measure."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from encoding_models import song_regressors as g
from synchrony import alice_comprehension as h, song_versions as c

N_LAGS, WIN_S, STREAM = 23, 0.8, 6


def rng_for(seed: int, key: str, sub_stream: int = 0) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(key.encode()), STREAM, sub_stream])


def perm_controls(trials: list[dict], raw_key: str, rng, n: int) -> list[dict]:
    """Surprisal values permuted among the words of each trial, timing kept, then band-passed."""
    ctrls = []
    for _ in range(n):
        cd = {}
        for j, tr in enumerate(trials):
            raw = np.zeros_like(tr[raw_key])
            raw[tr["idx"]] = rng.permutation(tr[raw_key][tr["idx"]])
            cd[j] = g.bz(raw[None])[:, tr["a"]:tr["a"] + tr["T"]]
        ctrls.append(cd)
    return ctrls


def canti_trials(specs: list[dict], a: int) -> list[dict]:
    trials = []
    for sp in specs:
        st, Y, v = sp["st"], sp["Y"], sp["v"]
        T = min(Y.shape[1], st["acoustic"].shape[1] - a)
        F = {k: st[k][:, a:a + T] for k in ("acoustic", "phonetic", "word", "surprisal") if k in st}
        trials.append({"F": F, "Y": Y[:, :T], "v": v[:T], "T": T, "a": a, "words": [i - a for i in st["word_idx"]],
                       "idx": np.asarray(st["word_idx"]), "raw_s": st["raw_surprisal"], "raw_e": sp.get("ent_raw"), "song": sp["song"]})
    return trials


def alice_trials(specs: list[dict], a: int) -> list[dict]:
    trials = []
    for sp in specs:
        st, Y, v = sp["st"], sp["Y"], sp["v"]
        T = min(Y.shape[1], st["acoustic"].shape[1] - a)
        F = {k: st[k][:, a:a + T] for k in ("acoustic", "word", "surprisal")}
        trials.append({"F": F, "Y": Y[:, :T], "v": v[:T], "T": T, "a": a, "words": [i - a for i in st["word_idx"]],
                       "idx": np.asarray(st["word_idx"]), "raw_s": st["raw_surprisal"], "seg": sp["seg"]})
    return trials


def canti_job_list(paths) -> list[tuple]:
    data = c.load_all(paths.derived_root / "cantisong_1to8", "1-8")
    words = {s: g.load_words(s) for s in c.song_ids()}
    a = int(c.START_S * 64)
    jobs = []
    for v in g.AUDIO:
        st = {s: g.stimulus(s, v, words[s], False) for s in c.song_ids()}
        subs = sorted({sub for (vv, s), d in data.items() if vv == v for sub in d})
        for sub in subs:
            songs = [s for s in c.song_ids() if (v, s) in data and sub in data[(v, s)] and data[(v, s)][sub][1].any()]
            if len(songs) < 6:
                continue
            specs = [{"st": st[s], "Y": data[(v, s)][sub][0], "v": data[(v, s)][sub][1], "song": s} for s in songs]
            jobs.append(("canti", f"canti|{v}|{sub}", specs, a))
    return jobs


def alice_job_list(paths) -> list[tuple]:
    words = pd.read_csv(paths.derived_root / "alice" / "words_gpt2.csv")
    st = {s: h.seg_stim(s, words, "gpt2_surprisal_bits") for s in h.SEGMENTS}
    data8, _ = h.load_all(paths.derived_root / "alice", "1-8")
    a = int(h.START_S * 64)
    jobs = []
    for sub in sorted(data8):
        segs = [s for s in h.SEGMENTS if s in data8[sub] and data8[sub][s][1].any()]
        if len(segs) < h.MIN_SEGMENTS:
            continue
        jobs.append(("alice", f"alice|speech|{sub}", [{"st": st[s], "Y": data8[sub][s][0], "v": data8[sub][s][1], "seg": s} for s in segs], a))
    return jobs
