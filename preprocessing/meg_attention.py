"""MEG preprocessing for the attention dataset.

Per listener and file: bad channels (robust z of log SD on a 1-40 Hz copy, |z| > 5, field interpolation); FastICA with 40
components on a 1-40 Hz copy at 200 Hz (cardiac from a synthetic ECG, ocular from a proxy EOG of the six most anterior
magnetometers, at most 3 removed); chapters resampled to 64 Hz and band-passed; epoch rejection with the EEG rule.
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from common.settings import dataset_root
from preprocessing.cnd_recordings import band_64
from preprocessing.epoch_rejection import reject
from preprocessing.epochs import Trial

ROOT = Path(__file__).parents[1]
DS = dataset_root("meg_attention")
TAB = ROOT / "results"
OUT = ROOT / "data" / "derived" / "meg_attention" / "meg"
QC = TAB / "qc"
FILES = {"LP": ("LP_speaker_attended", 1, 4), "HP": ("HP_speaker_attended", 3, 2)}
EDGE_S, EPOCH_S, FS = 2.0, 2.0, 64
SCALE = 1e15


def seed_int(key: str) -> int:
    return zlib.crc32(f"7|{key}".encode()) % (2 ** 31)


def chapter_lengths() -> dict[str, list[int]]:
    a = pd.read_csv(TAB / "meg_attention_audio.csv")
    return {f: a[a.book == f"audiobook{att}"].sort_values("chapter").n1000_ceil.astype(int).tolist()
            for f, (_, att, _) in FILES.items()}


def bad_channels(raw) -> tuple[list[str], dict]:
    r = raw.copy().filter(1.0, 40.0, verbose="error")
    lsd = np.log(r.get_data().std(1))
    med = np.median(lsd)
    mad = 1.4826 * np.median(np.abs(lsd - med))
    z = (lsd - med) / mad
    bads = [raw.ch_names[i] for i in np.flatnonzero(np.abs(z) > 5)]
    return bads, {raw.ch_names[i]: float(z[i]) for i in np.flatnonzero(np.abs(z) > 5)}


def anterior_mags(info, n: int = 6) -> list[int]:
    t = info["dev_head_t"]["trans"]
    pos = np.array([ch["loc"][:3] for ch in info["chs"]])
    head = pos @ t[:3, :3].T + t[:3, 3]
    return list(np.argsort(head[:, 1])[-n:])


def run_ica(raw, key: str) -> dict:
    import mne
    r = raw.copy().filter(1.0, 40.0, verbose="error").resample(200, verbose="error")
    ica = mne.preprocessing.ICA(n_components=40, method="fastica", random_state=seed_int(f"ica|{key}"), max_iter="auto",
                                verbose="error")
    ica.fit(r, verbose="error")
    out = {"n_components": int(ica.n_components_), "ecg_error": None, "eog_error": None}
    cand = []
    try:
        ecg_idx, ecg_scores = ica.find_bads_ecg(r, method="ctps", verbose="error")
        s = np.asarray(ecg_scores, float).ravel()
        z = (s - s.mean()) / (s.std() + 1e-12)
        cand += [{"component": int(i), "type": "cardiac", "score": float(s[i]), "z": float(z[i])} for i in ecg_idx]
    except Exception as e:                                                 # no detectable heartbeats
        out["ecg_error"] = f"{type(e).__name__}: {e}"
    try:
        ant = anterior_mags(r.info)
        proxy = mne.filter.filter_data(r.get_data(picks=ant).mean(0)[None], 200.0, None, 10.0, verbose="error")
        r.add_channels([mne.io.RawArray(proxy, mne.create_info(["EOGPROXY"], 200.0, ["eog"]), verbose="error")],
                       force_update_info=True)
        eog_idx, eog_scores = ica.find_bads_eog(r, ch_name="EOGPROXY", measure="zscore", threshold=3.0, verbose="error")
        s = np.asarray(eog_scores, float).ravel()
        z = (s - s.mean()) / (s.std() + 1e-12)
        cand += [{"component": int(i), "type": "ocular", "score": float(s[i]), "z": float(z[i])} for i in eog_idx]
        out["eog_proxy_channels"] = [raw.ch_names[i] for i in ant]
    except Exception as e:
        out["eog_error"] = f"{type(e).__name__}: {e}"
    best = {}
    for c in cand:
        if c["component"] not in best or abs(c["z"]) > abs(best[c["component"]]["z"]):
            best[c["component"]] = c
    ranked = sorted(best.values(), key=lambda c: -abs(c["z"]))
    out["flagged"] = ranked
    out["removed"] = ranked[:3]
    ica.exclude = [c["component"] for c in out["removed"]]
    ica.apply(raw, verbose="error")
    return out


def process_file(p: int, tag: str, lengths: list[int]) -> tuple[dict, dict]:
    import mne
    fname, att, ign = FILES[tag]
    raw = mne.io.read_raw_fif(DS / str(p) / fname / "data_meg.fif", preload=True, verbose="error")
    qc = {"file": fname, "n_times": raw.n_times}
    bads, bz = bad_channels(raw)
    qc["bad_channels"], qc["bad_channel_z"] = bads, bz
    if bads:
        raw.info["bads"] = bads
        raw.interpolate_bads(reset_bads=True, mode="accurate", origin="auto", verbose="error")
    qc["ica"] = run_ica(raw, f"{p}|{tag}")
    x = raw.get_data() * SCALE
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    qc["length_minus_audio"] = int(x.shape[1] - bounds[-1])
    chapters = {}
    for k in range(5):
        seg = x[:, bounds[k]: min(bounds[k + 1], x.shape[1])]
        y = resample_poly(seg, 8, 125, axis=1)
        edge = int(EDGE_S * FS)
        v = np.ones(y.shape[1], bool)
        v[:edge] = False
        v[-edge:] = False
        chapters[k + 1] = {"d18": band_64(y, FS, 1.0, 8.0), "d14": band_64(y, FS, 1.0, 4.0), "edge_ok": v}
    return chapters, qc


def process_listener(p: int) -> dict:
    target = OUT / f"P{p:02d}.npz"
    lengths = chapter_lengths()
    data, qc = {}, {"participant": p, "files": {}}
    for tag in FILES:
        ch, q = process_file(p, tag, lengths[tag])
        data[tag] = ch
        qc["files"][tag] = q
    trials = [Trial(subject=p, condition=tag, excerpt=k, label=tag, presentation_index=0, engagement=np.nan,
                    familiarity=np.nan, sfreq=FS, t0=0.0, data=data[tag][k]["d18"]) for tag in FILES for k in range(1, 6)]
    rej = reject(trials, [f"m{i}" for i in range(248)], EPOCH_S, seed_int(f"reject|{p}"), max_bad_fraction=1 / 3,
                 method="faster", z_threshold=3.0, channel_robust_z=5.0)
    arrays, chap_qc = {}, []
    n_ep = int(EPOCH_S * FS)
    for tr in trials:
        tag, k = tr.condition, tr.excerpt
        ok = rej.epoch_ok[tr.key]
        v = np.zeros(tr.data.shape[1], bool)
        v[: ok.size * n_ep] = np.repeat(ok, n_ep)
        kept = bool(rej.trial_ok[tr.key])
        valid = v & data[tag][k]["edge_ok"] & kept
        arrays[f"{tag}_{k}_d18"] = data[tag][k]["d18"].astype(np.float32)
        arrays[f"{tag}_{k}_d14"] = data[tag][k]["d14"].astype(np.float32)
        arrays[f"{tag}_{k}_valid"] = valid
        chap_qc.append({"file": tag, "chapter": k, "n_samples": int(tr.data.shape[1]), "n_epochs": int(ok.size),
                        "rejected_epochs": int((~ok).sum()), "rejected_fraction": float((~ok).mean()), "kept": kept,
                        "valid_fraction": float(valid.mean())})
    qc["chapters"] = chap_qc
    qc["dropped_chapters"] = [f"{c['file']}{c['chapter']}" for c in chap_qc if not c["kept"]]
    qc["reject_fraction"] = float(np.mean([c["rejected_fraction"] for c in chap_qc]))
    qc["included"] = all(sum(c["kept"] for c in chap_qc if c["file"] == f) >= 3 for f in FILES)
    tmp = OUT / f"P{p:02d}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, target)
    (QC / f"P{p:02d}.json").write_text(json.dumps(qc, indent=1, default=str), encoding="utf-8")
    return qc
