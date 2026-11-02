"""Transcription of the MEG attention audio.

Two recognisers transcribe the released audio in chunks of at most 25 s; chapters with more than 5 % word disagreement are
rejected, and disagreements are resolved by the forced-alignment likelihood under a German wav2vec2 model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

from common.settings import dataset_root
from encoding_models import german_text as T

ROOT = Path(__file__).parents[1]
AUDIO = dataset_root("meg_attention") / "Audio"
DER = ROOT / "data" / "derived" / "meg_attention"
OUT = ROOT / "results"
MODELS = {"whisper": ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"),
          "canary": ("nvidia/canary-1b-v2", "d455706339a6b32e1aa40f82c713a482a0c938e2"),
          "ctc": ("jonatasgrosman/wav2vec2-large-xlsr-53-german", "4b8a02957378d0f2da2ef74091156b032c485a89")}
FS = 16000
GATE = 0.05
TIE_NATS = 0.5
FRAME_S, HOP = 0.02, 320
WIN_S, CTX_S = 20.0, 2.0


def chapters() -> list[dict]:
    out = []
    for sub, stream in (("attended_speech", "attended"), ("ignored_speech", "ignored")):
        for f in sorted((AUDIO / sub).glob("*.wav")):
            book = int(f.stem[9])
            out.append({"stem": f.stem, "path": f, "stream": stream, "book": book, "chapter": int(f.stem[-1]),
                        "speaker": "LP" if book in (1, 2) else "HP"})
    return out


def load16k(path: Path) -> np.ndarray:
    sr, y = wavfile.read(path)
    y = np.asarray(y, np.float64)
    if y.dtype.kind != "f" or np.abs(y).max() > 1.5:
        y = y / 32768.0
    y = y.mean(1) if y.ndim > 1 else y
    return resample_poly(y, 160, 441).astype(np.float32)


def rms10(x: np.ndarray) -> np.ndarray:
    h = FS // 100
    n = len(x) // h
    return np.sqrt(np.mean(x[: n * h].reshape(n, h).astype(np.float64) ** 2, axis=1))


def speech_bounds(x: np.ndarray) -> tuple[int, int]:
    r = rms10(x)
    on = np.flatnonzero(r > np.percentile(r, 95) * 10 ** (-40 / 20))
    a = max(0, (on[0] - 30) * FS // 100)
    b = min(len(x), (on[-1] + 31) * FS // 100)
    return int(a), int(b)


def chunk_bounds(x: np.ndarray, lo_s: float = 15.0, hi_s: float = 25.0) -> list[tuple[int, int]]:
    r = rms10(x)
    sil = r < np.percentile(r, 95) * 10 ** (-35 / 20)
    n, s, out = len(r), 0, []
    while n - s > hi_s * 100:
        lo, hi = s + int(lo_s * 100), s + int(hi_s * 100)
        best, best_len, k = None, 0, lo
        while k < hi:
            if sil[k]:
                j = k
                while j < hi and sil[j]:
                    j += 1
                if j - k > best_len:
                    best, best_len = (k + j) // 2, j - k
                k = j
            else:
                k += 1
        cut = best if best is not None else lo + int(np.argmin(r[lo:hi]))
        out.append((s, cut))
        s = cut
    out.append((s, n))
    h = FS // 100
    return [(a * h, min(len(x), b * h)) for a, b in out]


def run_whisper():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
    rid, rev = MODELS["whisper"]
    proc = WhisperProcessor.from_pretrained(rid, revision=rev)
    model = WhisperForConditionalGeneration.from_pretrained(rid, revision=rev, dtype=torch.float16).to("cuda").eval()
    pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=proc.tokenizer,
                    feature_extractor=proc.feature_extractor, device="cuda", dtype=torch.float16)
    kw = {"language": "de", "task": "transcribe", "num_beams": 1, "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
          "compression_ratio_threshold": 1.35, "logprob_threshold": -1.0, "no_speech_threshold": 0.6}
    for ch in chapters():
        target = DER / "asr" / "whisper" / f"{ch['stem']}.json"
        if target.exists():
            continue
        x = load16k(ch["path"])
        a, b = speech_bounds(x)
        y = x[a:b]
        words, chunks = [], []
        for s, e in chunk_bounds(y):
            res = pipe(y[s:e].copy(), return_timestamps="word", generate_kwargs=kw)
            off = (a + s) / FS
            chunks.append({"start": off, "end": (a + e) / FS, "text": res["text"].strip()})
            for c in res["chunks"]:
                ts, te = c["timestamp"]
                te = ts if te is None else te
                parts = c["text"].split()
                for k, p in enumerate(parts):
                    words.append({"raw": p, "start": off + ts + (te - ts) * k / len(parts),
                                  "end": off + ts + (te - ts) * (k + 1) / len(parts)})
        target.write_text(json.dumps({"model": rid, "revision": rev, "generate_kwargs": {k: v for k, v in kw.items()},
                                      "bounds_s": [a / FS, b / FS], "chunks": chunks, "words": words},
                                     ensure_ascii=False, indent=1), encoding="utf-8")


def run_canary():
    from transformers import AutoProcessor, CanaryForConditionalGeneration
    rid, rev = MODELS["canary"]
    proc = AutoProcessor.from_pretrained(rid, revision=rev)
    model = CanaryForConditionalGeneration.from_pretrained(rid, revision=rev, dtype=torch.float16).to("cuda").eval()
    for ch in chapters():
        target = DER / "asr" / "canary" / f"{ch['stem']}.json"
        if target.exists():
            continue
        x = load16k(ch["path"])
        a, b = speech_bounds(x)
        y = x[a:b]
        chunks = []
        for s, e in chunk_bounds(y):
            inputs = proc.apply_transcription_request(y[s:e].copy(), source_language="de", punctuation=True).to("cuda")
            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.half()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, num_beams=1)
            chunks.append({"start": (a + s) / FS, "end": (a + e) / FS,
                           "text": proc.batch_decode(out, skip_special_tokens=True)[0].strip()})
        target.write_text(json.dumps({"model": rid, "revision": rev, "bounds_s": [a / FS, b / FS], "chunks": chunks},
                                     ensure_ascii=False, indent=1), encoding="utf-8")


def regions(ref: list[str], hyp: list[str]) -> list[list[int]]:
    """Non-matching runs as [ref_start, ref_end, hyp_start, hyp_end]; runs one matching word apart are merged."""
    import jiwer
    out = []
    for ck in jiwer.process_words(" ".join(ref), " ".join(hyp)).alignments[0]:
        if ck.type == "equal":
            continue
        r = [ck.ref_start_idx, ck.ref_end_idx, ck.hyp_start_idx, ck.hyp_end_idx]
        if out and r[0] - out[-1][1] <= 1 and r[2] - out[-1][3] <= 1:
            out[-1][1], out[-1][3] = r[1], r[3]
        else:
            out.append(r)
    return out


class CTC:
    def __init__(self):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        rid, rev = MODELS["ctc"]
        self.proc = Wav2Vec2Processor.from_pretrained(rid, revision=rev)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Wav2Vec2ForCTC.from_pretrained(rid, revision=rev).to(self.dev).eval()
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.model.config.pad_token_id

    def logprobs(self, x: np.ndarray) -> torch.Tensor:
        inp = self.proc(x, sampling_rate=FS, return_tensors="pt").input_values.to(self.dev)
        with torch.no_grad():
            return torch.log_softmax(self.model(inp).logits[0].float(), dim=-1)

    def target(self, words: list[str]) -> list[int]:
        return [self.vocab[c] for c in "|".join(words) if c in self.vocab]

    def score(self, lp: torch.Tensor, words: list[str]) -> float:
        tgt = self.target(words)
        if not tgt:
            return float(lp[:, self.blank].sum())
        reps = sum(1 for i in range(1, len(tgt)) if tgt[i] == tgt[i - 1])
        if len(tgt) + reps > lp.shape[0]:
            return float("nan")
        loss = torch.nn.functional.ctc_loss(lp[:, None, :], torch.tensor([tgt], device=lp.device),
                                            torch.tensor([lp.shape[0]]), torch.tensor([len(tgt)]),
                                            blank=self.blank, reduction="sum", zero_infinity=False)
        return float(-loss)


def emissions(ctc: CTC, x: np.ndarray) -> np.ndarray:
    """Frame log-probabilities (20 ms) over a whole chapter from 20 s windows with 2 s of context on each side."""
    n_frames = len(x) // HOP
    out = np.full((n_frames, len(ctc.vocab)), np.nan, np.float32)
    win, ctx = int(WIN_S * FS) // HOP * HOP, int(CTX_S * FS) // HOP * HOP
    for s in range(0, len(x), win):
        a, b = max(0, s - ctx), min(len(x), s + win + ctx)
        lp = ctc.logprobs(x[a:b]).cpu().numpy()
        f0 = a // HOP
        lo, hi = s // HOP, min(n_frames, (s + win) // HOP)
        j0, j1 = lo - f0, min(lp.shape[0], hi - f0)
        out[lo: lo + (j1 - j0)] = lp[j0:j1]
    last = np.flatnonzero(np.isfinite(out[:, 0]))[-1] + 1
    if not np.isfinite(out[:last]).all():
        raise RuntimeError("emission frames missing inside the chapter")
    return out[:last]


def viterbi(E: np.ndarray, tgt: list[int], blank: int) -> np.ndarray:
    """State index (into the blank-interleaved sequence) of the best CTC path at every frame."""
    ext = np.full(2 * len(tgt) + 1, blank)
    ext[1::2] = tgt
    S, Tn = len(ext), E.shape[0]
    skip = np.zeros(S, bool)
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    neg = -1e30
    alpha = np.full(S, neg)
    alpha[0], alpha[1] = E[0, ext[0]], E[0, ext[1]]
    bp = np.zeros((Tn, S), np.int8)
    for t in range(1, Tn):
        one = np.concatenate([[neg], alpha[:-1]])
        two = np.where(skip, np.concatenate([[neg, neg], alpha[:-2]]), neg)
        stack = np.vstack([alpha, one, two])
        k = stack.argmax(0)
        alpha = stack[k, np.arange(S)] + E[t, ext]
        bp[t] = k
    s = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    path = np.zeros(Tn, int)
    for t in range(Tn - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    return path


def word_times(path: np.ndarray, words: list[str], ctc: CTC) -> list[tuple[float, float]]:
    char_word = []
    for wi, w in enumerate(words):
        char_word += [wi] * len([c for c in w if c in ctc.vocab])
        if wi < len(words) - 1:
            char_word.append(-1)
    times = [[np.inf, -np.inf] for _ in words]
    for t, s in enumerate(path):
        if s % 2 == 1:
            wi = char_word[(s - 1) // 2]
            if wi >= 0:
                times[wi][0] = min(times[wi][0], t * FRAME_S)
                times[wi][1] = max(times[wi][1], (t + 1) * FRAME_S)
    return [(a, b) if np.isfinite(a) else (np.nan, np.nan) for a, b in times]


def resolve():
    ctc = CTC()
    rows, reg_rows = [], []
    for ch in chapters():
        w = json.loads((DER / "asr" / "whisper" / f"{ch['stem']}.json").read_text(encoding="utf-8"))
        c = json.loads((DER / "asr" / "canary" / f"{ch['stem']}.json").read_text(encoding="utf-8"))
        w_raw = [x["raw"] for x in w["words"]]
        c_raw = [t for ck in c["chunks"] for t in ck["text"].split()]
        ref, ref_i = T.normalise_words(w_raw)
        hyp, hyp_i = T.normalise_words(c_raw)
        regs = regions(ref, hyp)
        import jiwer
        m = jiwer.process_words(" ".join(ref), " ".join(hyp))
        n_edit = m.substitutions + m.deletions + m.insertions
        x = load16k(ch["path"])
        E = emissions(ctc, x)
        ct = np.array(word_times(viterbi(E, ctc.target(ref), ctc.blank), ref, ctc), float)
        ok = np.isfinite(ct[:, 0])
        k_ok = np.flatnonzero(ok)
        starts = np.interp(np.arange(len(ref)), k_ok, ct[ok, 0])
        ends = np.interp(np.arange(len(ref)), k_ok, ct[ok, 1])
        lp_full = torch.from_numpy(E).to(ctc.dev)
        la = ctc.score(lp_full, ref)
        choice = []
        for k, (rs, re_, hs, he) in enumerate(regs):
            rec = {"stem": ch["stem"], "region": k, "n_whisper": re_ - rs, "n_canary": he - hs,
                   "t_start": float(starts[rs - 1]) if rs > 0 else 0.0}
            lb = ctc.score(lp_full, ref[:rs] + hyp[hs:he] + ref[re_:])
            if not (np.isfinite(la) or np.isfinite(lb)):
                win = "unresolved"
            elif not np.isfinite(lb) or lb - la < TIE_NATS:
                win = "whisper"
            else:
                win = "canary"
            reg_rows.append(rec)
            choice.append(rec["winner"])
        resolved, src, times = [], [], []

        def emit(raws, idx, lo, hi, tag):
            for t in range(lo, hi):
                resolved.append(raws[idx[t]])
                src.append(tag)
                times.append([float(starts[t]), float(ends[t])] if tag == "whisper" else None)

        pos = 0
        for k, (rs, re_, hs, he) in enumerate(regs):
            emit(w_raw, ref_i, pos, rs, "whisper")
            if choice[k] == "canary":
                emit(c_raw, hyp_i, hs, he, "canary")
            else:
                emit(w_raw, ref_i, rs, re_, "whisper")
            pos = re_
        emit(w_raw, ref_i, pos, len(ref), "whisper")
        unresolved_words = sum(max(r["n_whisper"], r["n_canary"]) for r in reg_rows if r["stem"] == ch["stem"] and r["winner"] == "unresolved")
        rate = n_edit / max(1, len(ref))
        drop = rate > GATE and unresolved_words / max(1, len(ref)) > GATE
        (DER / "transcripts" / f"{ch['stem']}.json").write_text(json.dumps(
            {"stem": ch["stem"], "stream": ch["stream"], "speaker": ch["speaker"], "words": resolved, "source": src,
             "ctc_times_of_whisper_words": times, "dropped": bool(drop)}, ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append({"stem": ch["stem"], "stream": ch["stream"], "speaker": ch["speaker"], "book": ch["book"],
                     "chapter": ch["chapter"], "n_whisper": len(ref), "n_canary": len(hyp),
                     "substitutions": m.substitutions, "deletions": m.deletions, "insertions": m.insertions,
                     "disagreement": rate, "gate_pass": rate <= GATE, "n_regions": len(regs),
                     "regions_canary": choice.count("canary"), "regions_whisper": choice.count("whisper"),
                     "regions_unresolved": choice.count("unresolved"), "unresolved_word_fraction": unresolved_words / max(1, len(ref)),
                     "n_final": len(T.normalise_words(resolved)[0]), "dropped": drop})
    pd.DataFrame(rows).to_csv(OUT / "meg_attention_asr_agreement.csv", index=False)
    pd.DataFrame(reg_rows).to_csv(OUT / "meg_attention_asr_regions.csv", index=False)
