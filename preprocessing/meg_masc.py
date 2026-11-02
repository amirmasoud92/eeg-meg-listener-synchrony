"""MEG-MASC preprocessing.

Per listener and session: reference regression of the 208 gradiometers on the 16 reference magnetometers (betas on a
1-40 Hz copy at 200 Hz, applied to the unfiltered recording), bad channels with field interpolation, FastICA as in the
attention dataset, sound files cut at their onsets, resampled to 64 Hz and band-passed, epoch rejection pooled per
session (a story dropped above one third rejected; a session usable with at least three stories), and head position.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.spatial import cKDTree

from preprocessing.cnd_recordings import band_64
from preprocessing.epoch_rejection import reject
from preprocessing.epochs import Trial
from readers import meg_masc as C

OUT = C.DER / "meg"
QC = C.TAB / "qc"
EPOCH_S, SCALE = 2.0, 1e15


def audio_n1000(key: str) -> int:
    fs, x = wavfile.read(C.audio_path(key))
    return int(math.ceil(len(x) * 1000 / fs))


def points(path: Path) -> np.ndarray:
    return 1e-3 * np.loadtxt(path, comments="%", ndmin=2)              # FastSCAN millimetres, as MNE reads them


def robust_z_logsd(x: np.ndarray) -> np.ndarray:
    lsd = np.log(x.std(1) + 1e-300)
    med = np.median(lsd)
    mad = 1.4826 * np.median(np.abs(lsd - med))
    return (lsd - med) / (mad + 1e-12)


def head_metrics(raw, mrk_path: Path) -> dict:
    import mne
    from mne.io.constants import FIFF
    from mne.io.kit import read_mrk
    from mne.transforms import als_ras_trans, apply_trans, invert_transform
    head_dev = invert_transform(raw.info["dev_head_t"])
    dig = raw.info["dig"]
    hpi = np.array([d["r"] for d in dig if d["kind"] == FIFF.FIFFV_POINT_HPI])
    mrk = apply_trans(als_ras_trans, read_mrk(mrk_path))
    coil_err = np.linalg.norm(apply_trans(head_dev, hpi) - mrk, axis=1) if len(hpi) == len(mrk) else np.full(len(mrk), np.nan)
    shape = np.array([d["r"] for d in dig if d["kind"] == FIFF.FIFFV_POINT_EXTRA])
    sens = np.array([raw.info["chs"][i]["loc"][:3] for i in mne.pick_types(raw.info, meg=True, ref_meg=False)])
    dist = cKDTree(apply_trans(head_dev, shape)).query(sens)[0] if len(shape) else np.full(len(sens), np.nan)
    return {"coil_error_mm": [float(e * 1e3) for e in coil_err], "coil_error_max_mm": float(np.nanmax(coil_err) * 1e3),
            "head_origin_device_m": [float(v) for v in apply_trans(head_dev, np.zeros(3))],
            "head_to_sensor_mm": float(np.nanmean(dist) * 1e3), "n_head_points": int(len(shape))}


def apply_betas(raw, meg_names: list[str], ref_names: list[str], betas: np.ndarray, chunk: int = 200_000) -> None:
    """In place, exactly as MNE's EOGRegression.apply: data -= betas @ (references - their mean). MNE 1.13's
    regress_artifact cannot apply given betas (it passes picks_artifact positionally to EOGRegression), so the
    estimation uses regress_artifact and the application is done here."""
    pm = [raw.ch_names.index(n) for n in meg_names]
    pr = [raw.ch_names.index(n) for n in ref_names]
    mu = raw._data[pr].mean(-1, keepdims=True)
    for a in range(0, raw.n_times, chunk):
        b = min(a + chunk, raw.n_times)
        raw._data[pm, a:b] -= betas @ (raw._data[pr, a:b] - mu)


def anterior(raw, picks: list[int], n: int = 6) -> list[int]:
    t = raw.info["dev_head_t"]["trans"]
    pos = np.array([raw.info["chs"][i]["loc"][:3] for i in picks])
    head = pos @ t[:3, :3].T + t[:3, 3]
    return [picks[i] for i in np.argsort(head[:, 1])[-n:]]


def run_ica(raw, picks: list[int], key: str) -> dict:
    import mne
    r = raw.copy().pick(picks).filter(1.0, 40.0, verbose="error").resample(200, verbose="error")
    ica = mne.preprocessing.ICA(n_components=40, method="fastica", random_state=C.seed_int(f"ica|{key}"), max_iter="auto", verbose="error")
    ica.fit(r, verbose="error")
    out = {"n_components": int(ica.n_components_), "ecg_error": None, "eog_error": None}
    cand = []
    try:
        idx, scores = ica.find_bads_ecg(r, method="ctps", verbose="error")
        s = np.asarray(scores, float).ravel()
        z = (s - s.mean()) / (s.std() + 1e-12)
        cand += [{"component": int(i), "type": "cardiac", "score": float(s[i]), "z": float(z[i])} for i in idx]
    except Exception as e:                                                  # no detectable heartbeats
        out["ecg_error"] = f"{type(e).__name__}: {e}"
    try:
        ant = anterior(r, list(range(len(r.ch_names))))
        proxy = mne.filter.filter_data(r.get_data(picks=ant).mean(0)[None], 200.0, None, 10.0, verbose="error")
        r.add_channels([mne.io.RawArray(proxy, mne.create_info(["EOGPROXY"], 200.0, ["eog"]), verbose="error")], force_update_info=True)
        idx, scores = ica.find_bads_eog(r, ch_name="EOGPROXY", measure="zscore", threshold=3.0, verbose="error")
        s = np.asarray(scores, float).ravel()
        z = (s - s.mean()) / (s.std() + 1e-12)
        cand += [{"component": int(i), "type": "ocular", "score": float(s[i]), "z": float(z[i])} for i in idx]
        out["eog_proxy_channels"] = [r.ch_names[i] for i in ant]
    except Exception as e:
        out["eog_error"] = f"{type(e).__name__}: {e}"
    best = {}
    for c in cand:
        if c["component"] not in best or abs(c["z"]) > abs(best[c["component"]]["z"]):
            best[c["component"]] = c
    ranked = sorted(best.values(), key=lambda c: -abs(c["z"]))
    out["flagged"], out["removed"] = ranked, ranked[:3]
    ica.exclude = [c["component"] for c in out["removed"]]
    ica.apply(raw, verbose="error")
    return out


def process_recording(sub: str, ses: int, task: int) -> tuple[dict, dict]:
    import mne
    d, base = C.meg_dir(sub, ses), f"sub-{sub}_ses-{ses}"
    mrk = d / f"{base}_task-{task}_markers.mrk"
    raw = mne.io.read_raw_kit(d / f"{base}_task-{task}_meg.con", mrk=mrk, elp=points(d / f"{base}_acq-ELP_headshape.pos"),
                              hsp=points(d / f"{base}_acq-HSP_headshape.pos"), stim=None, preload=True, verbose="error")
    q = {"task": task, "story": C.STORY[task], "n_times": int(raw.n_times), "sfreq": float(raw.info["sfreq"])}
    q |= head_metrics(raw, mrk)
    meg = mne.pick_types(raw.info, meg=True, ref_meg=False)
    refs = mne.pick_types(raw.info, meg=False, ref_meg=True)
    q["n_meg"], q["n_ref"] = int(len(meg)), int(len(refs))
    # reference regression (betas on a 1-40 Hz copy at 200 Hz, applied to the unfiltered recording)
    tmp = raw.copy().pick(np.r_[meg, refs]).filter(1.0, 40.0, picks="all", verbose="error").resample(200, verbose="error")
    zr = robust_z_logsd(tmp.get_data(picks=[raw.ch_names[i] for i in refs]))
    good_refs = [raw.ch_names[i] for i, z in zip(refs, zr) if abs(z) <= 5]
    q["ref_excluded"] = [raw.ch_names[i] for i, z in zip(refs, zr) if abs(z) > 5]
    meg_names = [raw.ch_names[i] for i in meg]
    v0 = tmp.get_data(picks=meg_names).var(1)
    tmp, betas = mne.preprocessing.regress_artifact(tmp, picks=meg_names, picks_artifact=good_refs, proj=False, copy=False, verbose="error")
    q["ref_variance_removed_1_40"] = float(1 - tmp.get_data(picks=meg_names).var(1).sum() / v0.sum())
    apply_betas(raw, meg_names, good_refs, betas)
    # bad channels
    r = raw.copy().pick(meg_names).filter(1.0, 40.0, verbose="error")
    z = robust_z_logsd(r.get_data())
    bads = [meg_names[i] for i in np.flatnonzero(np.abs(z) > 5)]
    q["bad_channels"], q["bad_channel_z"] = bads, {meg_names[i]: float(z[i]) for i in np.flatnonzero(np.abs(z) > 5)}
    if bads:
        raw.info["bads"] = bads
        raw.interpolate_bads(reset_bads=True, mode="accurate", origin="auto", verbose="error")
    # ICA
    q["ica"] = run_ica(raw, list(mne.pick_channels(raw.ch_names, meg_names, ordered=True)), f"{sub}|{ses}|{task}")
    x = raw.get_data(picks=meg_names) * SCALE
    # sound files
    ev = C.events(sub, ses, task)
    snd = ev[ev.kind == "sound"].sort_values("onset")
    segs, seg_q = {}, []
    for _, s in snd.iterrows():
        key = C.sound_key(C.STORY[task], s.sound_id)
        n1000 = audio_n1000(key)
        a = int(round(float(s.onset) * 1000))
        seg = x[:, a: min(a + n1000, x.shape[1])]
        y = resample_poly(seg, 8, 125, axis=1)
        edge = int(C.EDGE_S * C.FS)
        v = np.ones(y.shape[1], bool)
        v[:edge] = v[-edge:] = False
        segs[key] = {"d18": band_64(y, C.FS, 1.0, 8.0), "d14": band_64(y, C.FS, 1.0, 4.0), "edge_ok": v}
        seg_q.append({"key": key, "start_sample": a, "n1000": n1000, "n1000_available": int(seg.shape[1]), "n64": int(y.shape[1])})
    q["sound_files"] = seg_q
    return segs, q


def process_session(sub: str, ses: int) -> dict:
    target = OUT / f"sub-{sub}_ses-{ses}.npz"
    if target.exists():
        return f"sub-{sub} ses-{ses} exists"
    segs, rec_q = {}, []
    for task in C.STORY:
        s, q = process_recording(sub, ses, task)
        segs |= s
        rec_q.append(q)
    trials = [Trial(subject=0, condition=key, excerpt=0, label=key, presentation_index=0, engagement=np.nan, familiarity=np.nan,
                    sfreq=C.FS, t0=0.0, data=segs[key]["d18"]) for key in segs]
    rej = reject(trials, [f"m{i}" for i in range(trials[0].data.shape[0])], EPOCH_S, C.seed_int(f"reject|{sub}|{ses}"),
                 max_bad_fraction=1.0, method="faster", z_threshold=3.0, channel_robust_z=5.0)
    n_ep = int(EPOCH_S * C.FS)
    arrays, stories = {}, []
    for task, story in C.STORY.items():
        keys = [k for k in segs if k.rsplit("_", 1)[0] == story]
        oks = [rej.epoch_ok[tr.key] for tr in trials if tr.condition in keys]
        n_all, n_bad = sum(o.size for o in oks), sum(int((~o).sum()) for o in oks)
        frac = n_bad / max(n_all, 1)
        kept = frac <= 1 / 3
        for tr in trials:
            if tr.condition not in keys:
                continue
            ok = rej.epoch_ok[tr.key]
            v = np.zeros(tr.data.shape[1], bool)
            v[: ok.size * n_ep] = np.repeat(ok, n_ep)
            arrays[f"{tr.condition}_d18"] = segs[tr.condition]["d18"].astype(np.float32)
            arrays[f"{tr.condition}_d14"] = segs[tr.condition]["d14"].astype(np.float32)
            arrays[f"{tr.condition}_valid"] = v & segs[tr.condition]["edge_ok"]
        stories.append({"task": task, "story": story, "sound_files": keys, "n_epochs": int(n_all), "rejected_epochs": int(n_bad),
                        "rejected_fraction": float(frac), "kept": bool(kept)})
    q = {"listener": sub, "session": ses, "recordings": rec_q, "stories": stories,
         "n_kept_stories": int(sum(s["kept"] for s in stories)), "usable": bool(sum(s["kept"] for s in stories) >= 3),
         "rejected_fraction": float(sum(s["rejected_epochs"] for s in stories) / max(sum(s["n_epochs"] for s in stories), 1)),
         "head_to_sensor_mm": float(np.mean([r["head_to_sensor_mm"] for r in rec_q])),
         "head_origin_device_m": [float(v) for v in np.mean([r["head_origin_device_m"] for r in rec_q], 0)],
         "coil_error_max_mm": float(max(r["coil_error_max_mm"] for r in rec_q))}
    tmp = OUT / f"sub-{sub}_ses-{ses}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, target)
    (QC / f"sub-{sub}_ses-{ses}.json").write_text(json.dumps(q, indent=1, default=str), encoding="utf-8")
    return q


def inclusion_table():
    rows, recs = [], []
    for sub in C.listeners():
        for ses in C.sessions(sub):
            f = C.TAB / "qc" / f"sub-{sub}_ses-{ses}.json"
            if not f.exists():
                raise SystemExit(f"missing QC: {f.name}")
            q = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"listener": sub, "session": ses, "usable": q["usable"], "n_kept_stories": q["n_kept_stories"],
                         "rejected_fraction": q["rejected_fraction"], "head_to_sensor_mm": q["head_to_sensor_mm"],
                         "head_x_m": q["head_origin_device_m"][0], "head_y_m": q["head_origin_device_m"][1],
                         "head_z_m": q["head_origin_device_m"][2], "coil_error_max_mm": q["coil_error_max_mm"],
                         "coil_flag": q["coil_error_max_mm"] > 10})
            for r, st in zip(q["recordings"], q["stories"]):
                recs.append({"listener": sub, "session": ses, "task": r["task"], "story": r["story"], "kept": st["kept"],
                             "rejected_fraction": st["rejected_fraction"], "bad_channels": len(r["bad_channels"]),
                             "ref_excluded": len(r["ref_excluded"]), "ref_variance_removed_1_40": r["ref_variance_removed_1_40"],
                             "ica_removed": len(r["ica"]["removed"]),
                             "ica_cardiac": sum(c["type"] == "cardiac" for c in r["ica"]["removed"]),
                             "ica_ocular": sum(c["type"] == "ocular" for c in r["ica"]["removed"]),
                             "coil_error_max_mm": r["coil_error_max_mm"], "head_to_sensor_mm": r["head_to_sensor_mm"],
                             "minutes": r["n_times"] / r["sfreq"] / 60})
    t = pd.DataFrame(rows)
    usable = t[t.usable].groupby("listener").session.nunique()
    t["two_session_set"] = t.listener.map(lambda s: usable.get(s, 0) == 2) & t.usable
    t["one_session_set"] = t.usable
    disp = {}
    for sub, g in t[t.two_session_set].groupby("listener"):
        a = g.sort_values("session")[["head_x_m", "head_y_m", "head_z_m"]].to_numpy()
        disp[sub] = float(np.linalg.norm(a[1] - a[0]) * 1e3)
    t["head_displacement_mm"] = t.listener.map(disp)
    t.to_csv(C.TAB / "meg_masc_inclusion.csv", index=False)
    r = pd.DataFrame(recs)
    r.to_csv(C.TAB / "meg_masc_qc_recordings.csv", index=False)
    s = {"listener_sessions": int(len(t)), "usable_sessions": int(t.usable.sum()),
         "two_session_listeners": int(t[t.two_session_set].listener.nunique()), "one_session_listeners": int(t[t.usable].listener.nunique()),
         "dropped_stories": int((~r.kept).sum()), "coil_flagged_sessions": int(t.coil_flag.sum()),
         "rejected_fraction_median": float(r.rejected_fraction.median()), "bad_channels_median": float(r.bad_channels.median()),
         "ref_variance_removed_median": float(r.ref_variance_removed_1_40.median()), "ica_removed_mean": float(r.ica_removed.mean()),
         "head_displacement_mm_median": float(np.median(list(disp.values()))) if disp else None,
         "head_to_sensor_mm_median": float(t.head_to_sensor_mm.median())}
    (C.TAB / "meg_masc_qc_summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
