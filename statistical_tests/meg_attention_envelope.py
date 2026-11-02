"""Envelope tracking against comprehension and synchrony in the MEG attention dataset, with partial correlations."""
from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd
from scipy import stats

from common.settings import PROJECT_ROOT
from readers import meg_attention as C

T7 = T8 = PROJECT_ROOT / "results"
N = 100_000


def seed(key: str) -> int:
    return zlib.crc32(f"8|{key}".encode())


def spearman_two_sided(x, y, key: str) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = stats.spearmanr(x, y).correlation
    g = np.random.default_rng(seed(key))
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    null = np.array([np.corrcoef(rx, g.permutation(ry))[0, 1] for _ in range(N)])
    return {"rho": float(r), "p_two_sided": float((1 + (np.abs(null) >= abs(r)).sum()) / (1 + N)), "n": int(x.size)}


def partial_two_sided(x, y, Z, key: str) -> dict:
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Zr = np.column_stack([np.ones(len(rx))] + [stats.rankdata(z) for z in Z])
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1])
    g = np.random.default_rng(seed(key))
    null = np.array([np.corrcoef(ex, g.permutation(ey))[0, 1] for _ in range(N)])
    return {"partial_rho": r, "p_two_sided": float((1 + (np.abs(null) >= abs(r)).sum()) / (1 + N)), "n": int(len(rx))}


def envelope_correlations():
    ck = pd.read_csv(T7 / "meg_attention_checks_listeners.csv").set_index("listener")
    sync = pd.read_csv(T7 / "meg_attention_sync_listeners.csv").set_index("listener")
    part = pd.read_csv(T7 / "meg_attention_participants.csv").set_index("participant")
    heads = pd.read_csv(T7 / "meg_attention_meg_headers.csv")
    head = heads[heads.file == "LP_speaker_attended"].set_index("participant")[["head_x_mm", "head_y_mm", "head_z_mm"]]
    rej = pd.Series({p: C.qc(p)["reject_fraction"] for p in range(1, 59)})
    d = pd.DataFrame({"comp": part.comprehension, "sync": sync.sync, "r_ign": ck.r_ignored_true, "r_att": ck.r_attended_true,
                      "reject": rej}).join(head).dropna()
    base = [d.reject, d.r_att, d.head_x_mm, d.head_y_mm, d.head_z_mm]
    res = {"n": int(len(d)),
           "rho_r_ignored_comprehension": spearman_two_sided(d.r_ign, d.comp, "rign_comp"),
           "rho_r_ignored_synchrony": spearman_two_sided(d.r_ign, d.sync, "rign_sync"),
           "rho_r_attended_comprehension": spearman_two_sided(d.r_att, d.comp, "ratt_comp"),
           "partial_listener_covariates": partial_two_sided(d.sync, d.comp, base, "partial_base"),
           "partial_plus_r_ignored": partial_two_sided(d.sync, d.comp, base + [d.r_ign], "partial_plus_rign"),
           "partial_r_ignored_only": partial_two_sided(d.sync, d.comp, [d.r_ign], "partial_rign_only"),
           "rho_r_attended_synchrony": spearman_two_sided(d.r_att, d.sync, "ratt_sync"),
           "partial_r_attended_only": partial_two_sided(d.sync, d.comp, [d.r_att], "partial_ratt_only"),
           "partial_without_r_attended": partial_two_sided(d.sync, d.comp, [d.reject, d.head_x_mm, d.head_y_mm, d.head_z_mm],
                                                           "partial_no_ratt"),
           "rho_synchrony_comprehension": spearman_two_sided(d.sync, d.comp, "sync_comp")}
    (T8 / "meg_attention_envelope.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
