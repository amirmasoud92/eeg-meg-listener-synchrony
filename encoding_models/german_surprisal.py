"""Word features for the MEG attention stimuli: GerPT2 surprisal, frequency, word class, duration and peak envelope."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss
import torch
from scipy.io import wavfile

from encoding_models import german_text as T, speech_envelope as tk, transcription as A
from readers import etard_release as etard

DER, OUT = A.DER, A.OUT
LMS = {"gerpt2": ("benjamin/gerpt2", "0093b45ef0d957cc545d872c208e66fc13861a0e")}
FS_ENV = 64


def surprisal_long(text: str, spans: list[tuple[int, int]], tok, model, win: int = 1024, stride: int = 512):
    enc = tok(text, return_offsets_mapping=True)
    ids = [tok.eos_token_id] + enc["input_ids"]
    offs = [(0, 0)] + enc["offset_mapping"]
    tok_s = np.full(len(ids), np.nan)
    start = 0
    while True:
        end = min(start + win, len(ids))
        with torch.no_grad():
            logp = torch.log_softmax(model(torch.tensor([ids[start:end]], device="cuda")).logits[0].float(), dim=-1)
        first = 1 if start == 0 else start + (win - stride)
        for k in range(max(first, start + 1), end):
            tok_s[k] = float(-logp[k - start - 1, ids[k]] / np.log(2))
        if end == len(ids):
            break
        start += stride
    s_out, n_out = [], []
    for a, b in spans:
        sel = [v for (o0, o1), v in zip(offs, tok_s) if o1 > a and o0 < b]
        s_out.append(float(np.nansum(sel)) if sel else np.nan)
        n_out.append(len(sel))
    return s_out, n_out, len(ids)


def envelope_1k(path: Path) -> np.ndarray:
    sr, y = wavfile.read(path)
    y = np.asarray(y, float)
    if y.dtype.kind != "f" or np.abs(y).max() > 1.5:
        y = y / 32768.0
    y = y.mean(1) if y.ndim > 1 else y
    return tk.broadband_envelope(y, sr, etard.load_config().section("envelope"))


def word_features():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    wt = pd.read_csv(DER / "meg_attention_word_times.csv")
    chs = {c["stem"]: c for c in A.chapters()}
    rows = []
    texts = {}
    for stem, g in wt.groupby("stem", sort=False):
        tr = json.loads((DER / "transcripts" / f"{stem}.json").read_text(encoding="utf-8"))
        raw = tr["words"]
        keep = sorted(set(g.raw_index))
        text, spans = "", {}
        for i in keep:
            if text:
                text += " "
            spans[i] = (len(text), len(text) + len(raw[i]))
            text += raw[i]
        texts[stem] = (text, spans)
    surp = {}
    for name, (rid, rev) in LMS.items():
        tok = AutoTokenizer.from_pretrained(rid, revision=rev)
        model = AutoModelForCausalLM.from_pretrained(rid, revision=rev, dtype=torch.float32).to("cuda").eval()
        for stem, (text, spans) in texts.items():
            idx = sorted(spans)
            s, n, ntok = surprisal_long(text, [spans[i] for i in idx], tok, model)
            surp[(name, stem)] = (dict(zip(idx, s)), dict(zip(idx, n)), ntok)
    for stem, g in wt.groupby("stem", sort=False):
        ch = chs[stem]
        tr = json.loads((DER / "transcripts" / f"{stem}.json").read_text(encoding="utf-8"))
        raw = tr["words"]
        env1k = envelope_1k(ch["path"])
        env64 = ss.resample_poly(env1k, FS_ENV, 1000)
        for r in g.itertuples():
            w = raw[r.raw_index]
            z, found = T.zipf(w)
            if np.isfinite(r.onset):
                a = int(round(r.onset * FS_ENV))
                b = max(int(round(r.offset * FS_ENV)), a + 1)
                peak = float(env64[a:b].max()) if a < env64.size else np.nan
            else:
                peak = np.nan
            rows.append({"stem": stem, "stream": ch["stream"], "speaker": ch["speaker"], "book": ch["book"], "chapter": ch["chapter"],
                         "word_index": r.word_index, "raw_index": r.raw_index, "word": w, "norm": T.normalise(w),
                         "onset": r.onset, "offset": r.offset, "duration": r.offset - r.onset, "aligned": r.aligned,
                         "surprisal_gerpt2": surp[("gerpt2", stem)][0][r.raw_index],
                         "n_tokens_gerpt2": surp[("gerpt2", stem)][1][r.raw_index],
                         "zipf": z, "zipf_found": found, "function": T.is_function(w), "peak_env": peak,
                         "source": tr["source"][r.raw_index]})
    d = pd.DataFrame(rows)
    d.to_csv(DER / "meg_attention_words.csv", index=False)
    d.drop(columns=["word", "norm", "raw_index"]).to_csv(OUT / "meg_attention_word_features.csv", index=False)
