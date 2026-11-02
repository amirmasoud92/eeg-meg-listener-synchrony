"""Word-level gain in the speech-in-noise data, audiovisual against audio-only within listeners.

Per listener and condition: leave-one-passage-out ridge, lags 0-688 ms, content-word onsets and frequency-scaled
impulses as the base (no acoustics are released), surprisal as the tested regressor, and 100 within-passage permutations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.settings import dataset_root

ROOT = Path(__file__).parents[1]
SIN_TXT = ROOT / "data" / "derived" / "speech_in_noise_texts"
WORDS = ROOT / "data" / "derived" / "speech_in_noise_words"
STIM = dataset_root("speech_in_noise") / "Stimuli"
N_PERM, N_LAGS, WIN_S, MIN_PASSAGES = 100, 23, 0.8, 10
CONDS = ("a", "av")


def passage_words():
    import pandas as pd
    import scipy.io as sio
    from encoding_models import english_lexicon as X
    align = json.loads((SIN_TXT / "word_map.json").read_text())
    for k in range(1, 16):
        m = sio.loadmat(STIM / f"Run{k}.mat", squeeze_me=True)
        words = [str(w) for w in np.atleast_1d(m["wordVec"])]
        al = align[str(k)]
        clean = (SIN_TXT / f"clean_{al['source']}.txt").read_text(encoding="utf-8")
        spans = json.loads((SIN_TXT / f"spans_{al['source']}.json").read_text())
        mp = {int(i): t for i, t in al["map"].items()}
        first_char = spans[min(mp.values())][0]
        last_char = spans[max(mp.values())][1]
        start = first_char                       # passages start mid-sentence; context = what was heard
        rows = []
        for i, w in enumerate(words):
            t = mp.get(i)
            a_, b_ = (spans[t][0] - start, spans[t][1] - start) if t is not None else (np.nan, np.nan)
            rows.append({"i": i, "word": w, "onset": float(np.atleast_1d(m["onset_time"])[i]), "offset": float(np.atleast_1d(m["offset_time"])[i]),
                         "matched": t is not None, "char_start": a_, "char_end": b_, "zipf": X.zipf(w)[0], "function": X.is_function(w)})
        pd.DataFrame(rows).to_csv(WORDS / f"passage{k:02d}.csv", index=False)
        (WORDS / f"passage{k:02d}_context.txt").write_text(clean[start:last_char], encoding="utf-8")


def passage_surprisal():
    import pandas as pd
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from encoding_models.surprisal import surprisal_long
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    torch.manual_seed(0)
    for k in range(1, 16):
        d = pd.read_csv(WORDS / f"passage{k:02d}.csv")
        text = (WORDS / f"passage{k:02d}_context.txt").read_text(encoding="utf-8")
        mk = d.matched.to_numpy()
        spans = [(int(a), int(b)) for a, b in zip(d.char_start[mk], d.char_end[mk])]
        s = np.full(len(d), np.nan)
        s[mk] = surprisal_long(text, spans, tok, model)
        s[~np.isfinite(s)] = np.nanmedian(s)
        d["surprisal_bits"] = s
        d.to_csv(WORDS / f"passage{k:02d}.csv", index=False)


def _prepare(sub, labels, cfg2, seed, cache):
    from synchrony import speech_in_noise as d01
    d01.BANDS = {"1-8": (1.0, 8.0)}                 # set inside the worker
    return d01.prepare_one(sub, labels, cfg2, seed, cache)


def prepare_passages(n_jobs: int = 6):
    from joblib import Parallel, delayed
    from synchrony import speech_in_noise as d01
    from common.settings import load_analysis, load_paths
    paths, cfg = load_paths(), load_analysis()
    cache = paths.derived_root / "speech_in_noise_1to8"
    labels = d01.labels_128(paths)
    Parallel(n_jobs=n_jobs)(delayed(_prepare)(s, labels, cfg.section("preprocessing"), cfg.project.seed, cache) for s in d01.SUBJECTS)


def passage_regressors(k: int, n: int):
    """Raw impulses (onset, Zipf z-scored within passage, surprisal) on the 64 Hz grid of the full passage."""
    import pandas as pd
    d = pd.read_csv(WORDS / f"passage{k:02d}.csv")
    idx = (d.onset.to_numpy() * 64).round().astype(int)
    ok = (idx >= 0) & (idx < n)
    d, idx = d[ok], idx[ok]
    z = d.zipf.to_numpy()
    z = (z - z.mean()) / (z.std() + 1e-12)
    on, zi, su = np.zeros(n), np.zeros(n), np.zeros(n)
    np.add.at(on, idx, 1.0)
    np.add.at(zi, idx, z)
    np.add.at(su, idx, d.surprisal_bits.to_numpy())
    return on, zi, su, idx


def listener_job(sub: int, cache: Path, out_dir: Path, seed: int) -> str:
    import zlib
    from encoding_models import song_regressors as g
    import ridge_encoding as E
    from encoding_models import ridge_folds as WF
    from synchrony import speech_in_noise as d01
    f = out_dir / f"sub{sub:02d}.npz"
    if f.exists():
        return f.name
    z = np.load(cache / f"sub{sub:02d}.npz")
    a64 = int(d01.START_S * 64)
    res = {}
    for cond in CONDS:
        trials = []
        for p in d01.PASSAGES:
            kd, kv = f"1-8|{cond}|{p}", f"valid|{cond}|{p}"
            if kd not in z.files or not z[kv].any():
                continue
            Y, v = z[kd].astype(float), z[kv].astype(bool)
            T = Y.shape[1]
            n = a64 + T
            on, zi, su, idx = passage_regressors(p, n)
            F = {"word": g.bz(on[None])[:, a64:a64 + T], "zipf": g.bz(zi[None])[:, a64:a64 + T], "surprisal": g.bz(su[None])[:, a64:a64 + T]}
            trials.append({"F": F, "Y": Y, "v": v[:T], "T": T, "a": a64, "words": [i - a64 for i in idx], "idx": idx, "raw_s": su, "passage": p})
        if len(trials) < MIN_PASSAGES:
            res[cond] = None
            continue
        rng = np.random.default_rng([seed, zlib.crc32(f"sin|{cond}|{sub}".encode()), 6, 0])
        ctrls = []
        for _ in range(N_PERM):
            cd = {}
            for j, tr in enumerate(trials):
                raw = np.zeros_like(tr["raw_s"])
                raw[tr["idx"]] = rng.permutation(tr["raw_s"][tr["idx"]])
                cd[j] = g.bz(raw[None])[:, tr["a"]:tr["a"] + tr["T"]]
            ctrls.append(cd)
        det, lam = WF.delta_r_detail(trials, ["word", "zipf"], "surprisal", ctrls, E.event_masks(trials, "words", WIN_S), N_LAGS)
        res[cond] = {"pooled": det["pooled"], "lam": lam, "n_passages": len(trials), "null": det["null_pooled"].tolist()}
    np.savez(f, meta=json.dumps({"sub": sub, **{c: (None if r is None else {k: v for k, v in r.items() if k != "null"}) for c, r in res.items()}}),
             **{f"null_{c}": np.array(r["null"]) for c, r in res.items() if r is not None})
    return f.name


def listener_gains(n_jobs: int = 4):
    from joblib import Parallel, delayed
    from synchrony import speech_in_noise as d01
    from common.settings import load_analysis, load_paths
    paths, cfg = load_paths(), load_analysis()
    cache = paths.derived_root / "speech_in_noise_1to8"
    out_dir = paths.derived_root / "speech_in_noise_word_level"
    Parallel(n_jobs=n_jobs)(delayed(listener_job)(s, cache, out_dir, cfg.project.seed) for s in d01.SUBJECTS)


def av_minus_audio_test():
    import pandas as pd
    from common.settings import load_paths
    from statistical_tests.inference import summarise
    paths = load_paths()
    out_dir = paths.derived_root / "speech_in_noise_word_level"
    rows = []
    for f in sorted(out_dir.glob("sub*.npz")):
        m = json.loads(str(np.load(f)["meta"]))
        rows.append({"listener": m["sub"], **{f"dr_{c}": (m[c]["pooled"] if m[c] else np.nan) for c in CONDS},
                     **{f"n_{c}": (m[c]["n_passages"] if m[c] else 0) for c in CONDS}})
    df = pd.DataFrame(rows)
    df.to_csv(paths.results_root / "speech_in_noise_word_level_listeners.csv", index=False)
    pk = lambda x: {k: v for k, v in summarise(np.asarray(x, float), "greater").items() if k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}
    both = df.dropna(subset=["dr_a", "dr_av"])          # a listener enters with >= 10 valid passages in both
    res = {"primary_av_minus_a": pk(both.dr_av - both.dr_a), "secondary_a": pk(both.dr_a), "secondary_av": pk(both.dr_av),
           "n_listeners_both": int(len(both)), "n_listeners_total": int(len(df))}
    (paths.results_root / "speech_in_noise_word_level.json").write_text(json.dumps(res, indent=2, default=float))
