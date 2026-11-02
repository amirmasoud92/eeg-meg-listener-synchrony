"""Regional spatial filters per listener: eLORETA, LCMV and the geometry-based sensor selection."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from common.settings import load_analysis, load_paths
from network_states.source_model import TemplateModel, region_names
from preprocessing.aalborg_trials import CleanDataset
from readers.aalborg_release import load_channel_names

def sensor_selection(paths) -> tuple[np.ndarray, dict]:
    ch = load_channel_names(paths.channel_list)
    m = TemplateModel.build(ch, 512.0, paths.mne_data, paths.derived_root / "source_model")
    assert m.src[0]["coord_frame"] == 4  # head frame, same as the montage
    rr = np.vstack([s["rr"][s["vertno"]] for s in m.src])
    pos = m.info.get_montage().get_positions()["ch_pos"]
    E = np.array([pos[c] for c in ch])
    c0 = least_squares(lambda p: np.linalg.norm(E - p[:3], axis=1) - p[3], np.r_[E.mean(0), 0.09]).x[:3]
    names = region_names()
    cent = np.array([rr[v].mean(0) for v in m.label_vertices])
    u = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    cos = u(cent - c0) @ u(E - c0).T
    used, choice = set(), {}
    for r in np.argsort(-cos.max(1)):
        i = [j for j in np.argsort(-cos[r]) if j not in used][0]
        choice[r] = i
        used.add(i)
    W = np.zeros((len(names), len(ch)))
    for r, i in choice.items():
        W[r, i] = 1.0
    return W, {names[r]: ch[choice[r]] for r in range(len(names))}


def regional_filters():
    paths, cfg = load_paths(), load_analysis()
    c3 = cfg.section("source_model")
    out_d = paths.derived_root / "source_model"
    out_q = paths.results_root / "qc"
    ch_names = load_channel_names(paths.channel_list)
    sfreq = cfg.dataset.preprocessed_sfreq
    model = TemplateModel.build(ch_names, sfreq, paths.mne_data, out_d, mindist_mm=c3["mindist_mm"])
    names = region_names()
    K = model.eloreta_kernel(c3["eloreta_lambda2"], c3["eloreta_loose"], c3["eloreta_depth"])

    ds = CleanDataset(paths, cfg)
    for s in ds.subjects:
        trials = ds.participant(s)
        cov, n = ds.pooled_covariance(trials)
        ev = np.linalg.eigvalsh(cov)[::-1]
        rank = int((ev / ev[0] > c3["rank_tol"]).sum())
        W_e = model.regional_filters(K, cov)
        K_l = model.lcmv_kernel(cov, n, c3["lcmv_reg"], rank, c3["lcmv_weight_norm"], c3["lcmv_pick_ori"])
        W_l = model.regional_filters(K_l, cov)
        np.savez(out_d / f"filters_S{s}.npz", eloreta=W_e, lcmv=W_l, regions=np.array(names), rank=rank)
