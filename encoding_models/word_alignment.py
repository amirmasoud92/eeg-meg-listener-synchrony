"""Word timings for the MEG attention audio from the Montreal Forced Aligner with the pretrained German models."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from encoding_models import german_text as T, transcription as A

DER = A.DER
OUT = A.OUT
MFA_ENV = Path(os.environ["MFA_ENV"])                # conda environment with the Montreal Forced Aligner
MFA_WORK = Path(os.environ.get("MFA_WORK", "mfa_work"))
CORPUS = MFA_WORK / "corpus"
MFA_TMP = MFA_WORK / "aligned"
MFA_OUT = DER / "mfa" / "aligned"
MFA_CONFIG = {"beam": 100, "retry_beam": 400}
GAP_S, MIN_UTT_S, MAX_UTT_S = 0.25, 6.0, 25.0
ALIGN_GATE = 0.95
emissions, viterbi, word_times = A.emissions, A.viterbi, A.word_times


def utterances(times: list[tuple[float, float]], dur: float) -> list[tuple[float, float, int, int]]:
    on = np.array([a for a, _ in times])
    off = np.array([b for _, b in times])
    ok = np.isfinite(on)
    idx = np.flatnonzero(ok)
    utt, start_i = [], idx[0]
    t0 = max(0.0, on[start_i] - GAP_S)
    for p in range(len(idx) - 1):
        i, j = idx[p], idx[p + 1]
        gap = on[j] - off[i]
        length = off[i] - on[start_i]
        if (gap >= GAP_S and length >= MIN_UTT_S) or length >= MAX_UTT_S:
            if length >= MAX_UTT_S and gap < GAP_S:
                seg = idx[(idx >= start_i) & (idx <= i)]
                gaps = on[seg[1:]] - off[seg[:-1]]
                q = int(np.argmax(gaps))
                i, j = seg[q], seg[q + 1]
            mid = (off[i] + on[j]) / 2
            utt.append((t0, mid, start_i, j))
            start_i, t0 = j, mid
    utt.append((t0, min(dur, off[idx[-1]] + GAP_S), start_i, len(times)))
    return utt


def textgrid(path: Path, tier: str, dur: float, utts: list[tuple[float, float, str]]):
    iv, t = [], 0.0
    for a, b, txt in utts:
        if a > t + 1e-6:
            iv.append((t, a, ""))
        iv.append((a, b, txt))
        t = b
    if t < dur - 1e-6:
        iv.append((t, dur, ""))
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {dur}", "tiers? <exists>",
             "size = 1", "item []:", "    item [1]:", '        class = "IntervalTier"', f'        name = "{tier}"',
             "        xmin = 0", f"        xmax = {dur}", f"        intervals: size = {len(iv)}"]
    for k, (a, b, txt) in enumerate(iv, 1):
        lines += [f"        intervals [{k}]:", f"            xmin = {a}", f"            xmax = {b}", f'            text = "{txt}"']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ctc_alignment():
    ctc = A.CTC()
    for ch in A.chapters():
        tr = json.loads((DER / "transcripts" / f"{ch['stem']}.json").read_text(encoding="utf-8"))
        if tr["dropped"]:
            continue
        norm, raw_i = T.normalise_words(tr["words"])
        x = A.load16k(ch["path"])
        E = emissions(ctc, x)
        tgt = ctc.target(norm)
        path = viterbi(E, tgt, ctc.blank)
        times = word_times(path, norm, ctc)
        dur = len(x) / A.FS
        utts = [(float(a), float(b), int(i), int(j)) for a, b, i, j in utterances(times, dur)]
        (DER / "ctc" / f"{ch['stem']}.json").write_text(json.dumps(
            {"norm": norm, "raw_index": raw_i, "times": times, "utterances": utts}, ensure_ascii=False), encoding="utf-8")
        spk = CORPUS / ch["speaker"]
        wavfile.write(spk / f"{ch['stem']}.wav", A.FS, np.clip(x * 32767, -32768, 32767).astype(np.int16))
        textgrid(spk / f"{ch['stem']}.TextGrid", ch["speaker"], dur,
                 [(a, b, " ".join(norm[i:j])) for a, b, i, j in utts])


def mfa_alignment():
    env = os.environ.copy()
    e = str(MFA_ENV)
    env["PATH"] = os.pathsep.join([e, e + r"\Library\mingw-w64\bin", e + r"\Library\usr\bin", e + r"\Library\bin",
                                   e + r"\Scripts", env["PATH"]])
    env["PYTHONIOENCODING"] = "utf-8"
    import shutil
    cfg = MFA_WORK / "align_config.yaml"
    cfg.write_text("".join(f"{k}: {v}\n" for k, v in MFA_CONFIG.items()), encoding="utf-8")
    cmd = [str(MFA_ENV / "python.exe"), "-m", "montreal_forced_aligner", "align", str(CORPUS), "german_mfa", "german_mfa",
           str(MFA_TMP), "--config_path", str(cfg), "--g2p_model_path", "german_mfa", "--output_format", "json",
           "--no_tokenization", "--clean", "--overwrite", "-j", "4"]
    subprocess.run(cmd, env=env, check=True)
    if MFA_OUT.exists():
        shutil.rmtree(MFA_OUT)
    shutil.copytree(MFA_TMP, MFA_OUT)
    shutil.copytree(CORPUS, DER / "mfa" / "corpus", dirs_exist_ok=True)


def _mfa_words(stem: str, speaker: str) -> list[tuple[float, float, str]]:
    f = MFA_OUT / speaker / f"{stem}.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text(encoding="utf-8"))
    tiers = d["tiers"]
    tier = tiers.get("words") or next(v for k, v in tiers.items() if k.endswith("words"))
    return [(float(a), float(b), str(lab)) for a, b, lab in tier["entries"] if str(lab).strip() and str(lab) not in ("<eps>", "sil", "sp")]


def collect_word_times():
    rows, summ = [], []
    for ch in A.chapters():
        tr = json.loads((DER / "transcripts" / f"{ch['stem']}.json").read_text(encoding="utf-8"))
        if tr["dropped"]:
            continue
        c = json.loads((DER / "ctc" / f"{ch['stem']}.json").read_text(encoding="utf-8"))
        norm, raw_i, ctimes = c["norm"], c["raw_index"], c["times"]
        mw = _mfa_words(ch["stem"], ch["speaker"])
        import difflib
        sm = difflib.SequenceMatcher(a=norm, b=[w[2] for w in mw], autojunk=False)
        on = np.full(len(norm), np.nan)
        off = np.full(len(norm), np.nan)
        for blk in sm.get_matching_blocks():
            for k in range(blk.size):
                a, b, _ = mw[blk.b + k]
                if b > a:
                    on[blk.a + k], off[blk.a + k] = a, b
        aligned = np.isfinite(on)
        con = np.array([t[0] if t[0] is not None else np.nan for t in ctimes], float)
        diff = np.abs(on - con)
        for k in range(len(norm)):
            rows.append({"stem": ch["stem"], "stream": ch["stream"], "speaker": ch["speaker"], "book": ch["book"],
                         "chapter": ch["chapter"], "word_index": k, "raw_index": raw_i[k], "onset": on[k], "offset": off[k],
                         "ctc_onset": con[k], "ctc_offset": ctimes[k][1], "aligned": bool(aligned[k])})
        summ.append({"stem": ch["stem"], "stream": ch["stream"], "speaker": ch["speaker"], "n_words": len(norm),
                     "n_mfa_words": len(mw), "aligned_fraction": float(aligned.mean()), "gate_pass": bool(aligned.mean() >= ALIGN_GATE),
                     "median_abs_onset_mfa_vs_ctc_ms": float(np.nanmedian(diff) * 1e3),
                     "p95_abs_onset_mfa_vs_ctc_ms": float(np.nanpercentile(diff, 95) * 1e3),
                     "share_abs_onset_diff_over_100ms": float(np.nanmean(diff > 0.1)),
                     "n_utterances": len(c["utterances"])})
    w = pd.DataFrame(rows)
    w.to_csv(DER / "meg_attention_word_times.csv", index=False)
    w.drop(columns=["raw_index"]).to_csv(OUT / "meg_attention_word_times.csv", index=False)
    pd.DataFrame(summ).to_csv(OUT / "meg_attention_alignment_summary.csv", index=False)
