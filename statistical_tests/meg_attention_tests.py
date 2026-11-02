"""Listener-level tests of the MEG attention dataset.

Monte Carlo rule for the word-level gain, Spearman tests against comprehension, the difference between correlations
(listener bootstrap, 10^4), Freedman-Lane partial correlations and inverse-variance pooling with Alice.
"""
from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd
from scipy import stats

from readers import meg_attention as C
from statistical_tests.inference import dz_ci, jzs_bf10, partial_test, sign_flip_test

N_FLIP, N_PERM, N_BOOT = 100_000, 100_000, 10_000
VARIANTS = {"primary": 500}
SUB = {"all": 0, "LP": 1, "HP": 2}


def seed(key: str) -> int:
    return zlib.crc32(f"7|{key}".encode())


def one_sample(x, alternative: str, key: str) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    p, _ = sign_flip_test(x, alternative, n_perm=N_FLIP, rng=C.rng(f"p7|signflip|{key}"))
    lo, hi = dz_ci(x)
    out = {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "dz": float(x.mean() / x.std(ddof=1)),
           "dz_ci95": [float(lo), float(hi)], "p": float(p), "alternative": alternative, "n_positive": int((x > 0).sum())}
    out["bf01_two_sided"] = float(1.0 / jzs_bf10(x, alternative="two-sided"))
    return out


def spearman_perm(x, y, key: str, two_sided: bool = False) -> dict:
    """One-sided permutation of ranks, with an optional two-sided p."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = stats.spearmanr(x, y).correlation
    g = np.random.default_rng(seed(key))
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    null = np.array([np.corrcoef(rx, g.permutation(ry))[0, 1] for _ in range(N_PERM)])
    p = (1 + (np.abs(null) >= abs(r)).sum()) / (1 + N_PERM) if two_sided else (1 + (null >= r).sum()) / (1 + N_PERM)
    return {"rho": float(r), "p": float(p), "n": int(x.size), "alternative": "two-sided" if two_sided else "greater"}


def fisher_ci(r: float, n: int) -> list[float]:
    z, se = np.arctanh(r), np.sqrt(1.06 / (n - 3))
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def delta_rho_boot(comp, sync, dr, key: str) -> tuple[float, np.ndarray]:
    comp, sync, dr = (np.asarray(v, float) for v in (comp, sync, dr))
    g = C.rng(key)
    n = comp.size
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ix = g.integers(0, n, n)
        out[b] = stats.spearmanr(comp[ix], sync[ix]).correlation - stats.spearmanr(comp[ix], dr[ix]).correlation
    obs = stats.spearmanr(comp, sync).correlation - stats.spearmanr(comp, dr).correlation
    return float(obs), out


def qc_tables() -> tuple[pd.DataFrame, dict]:
    files, ica, chap = [], [], []
    for p in range(1, 59):
        q = C.qc(p)
        for tag, v in q["files"].items():
            files.append({"listener": p, "file": tag, "n_bad_channels": len(v["bad_channels"]),
                          "bad_channels": ";".join(v["bad_channels"]), "n_ica_flagged": len(v["ica"]["flagged"]),
                          "n_ica_removed": len(v["ica"]["removed"]), "ecg_error": v["ica"]["ecg_error"],
                          "eog_error": v["ica"]["eog_error"], "length_minus_audio": v["length_minus_audio"]})
            for c in v["ica"]["removed"]:
                ica.append({"listener": p, "file": tag, **c})
        for c in q["chapters"]:
            chap.append({"listener": p, **c})
    F, I, H = pd.DataFrame(files), pd.DataFrame(ica), pd.DataFrame(chap)
    F.to_csv(C.TAB / "meg_attention_qc_files.csv", index=False)
    I.to_csv(C.TAB / "meg_attention_qc_ica_removed.csv", index=False)
    H.to_csv(C.TAB / "meg_attention_qc_chapters.csv", index=False)
    lst = pd.DataFrame([{"listener": p, "included": C.qc(p)["included"], "reject_fraction": C.qc(p)["reject_fraction"],
                         "dropped": ";".join(C.qc(p)["dropped_chapters"])} for p in range(1, 59)])
    summ = {"files_with_bad_channels": int((F.n_bad_channels > 0).sum()), "bad_channels_total": int(F.n_bad_channels.sum()),
            "bad_channels_max_per_file": int(F.n_bad_channels.max()),
            "ica_removed_per_file": F.n_ica_removed.value_counts().sort_index().to_dict(),
            "ica_removed_by_type": I.type.value_counts().to_dict() if len(I) else {},
            "ecg_errors": int(F.ecg_error.notna().sum()), "eog_errors": int(F.eog_error.notna().sum()),
            "rejected_fraction_mean": float(H.rejected_fraction.mean()), "rejected_fraction_max": float(H.rejected_fraction.max()),
            "chapters_dropped": int((~H.kept).sum()), "listeners_included": int(lst.included.sum()),
            "listeners_excluded": lst.loc[~lst.included, "listener"].tolist()}
    return lst, summ


def load_A(listeners: list[int]) -> dict:
    out = {}
    for p in listeners:
        with np.load(C.DER / "A" / f"P{p:02d}.npz") as z:
            if "excluded" in z.files:
                continue
            out[p] = {k: z[k] for k in z.files}
    return out


def mc_rule(A: dict, variant: str, stream: str) -> tuple[int, list[dict]]:
    trail = []
    n_max = VARIANTS[variant]
    for n in range(100, n_max + 1, 100):
        dr, se = [], []
        for d in A.values():
            null = d[f"{variant}_{stream}_rperm"][:n, 0, :].mean(1)
            dr.append(d[f"{variant}_{stream}_rtrue"][0].mean() - null.mean())
            se.append(null.std(ddof=1) / np.sqrt(n))
        ratio = float(np.max(se) / np.std(dr, ddof=1))
        trail.append({"n": n, "max_ratio": ratio})
        if ratio <= 0.1:
            return n, trail
    return n_max, trail


def delta_r(A: dict, variant: str, stream: str, n: int, sub: str) -> pd.Series:
    i = SUB[sub]
    return pd.Series({p: float((d[f"{variant}_{stream}_rtrue"][i] - d[f"{variant}_{stream}_rperm"][:n, i].mean(0)).mean())
                      for p, d in A.items()})


def listener_tests():
    res = {}
    lst, res["qc"] = qc_tables()
    inc = lst.loc[lst.included, "listener"].tolist()
    part = pd.read_csv(C.TAB / "meg_attention_participants.csv").set_index("participant")
    heads = pd.read_csv(C.TAB / "meg_attention_meg_headers.csv")
    head = heads[heads.file == "LP_speaker_attended"].set_index("participant")[["head_x_mm", "head_y_mm", "head_z_mm"]]

    ck = pd.read_csv(C.TAB / "meg_attention_checks_listeners.csv")
    ck = ck[ck.included.astype(bool)].set_index("listener")
    chk = {"positive_control": one_sample(ck.r_attended_true - ck.r_ignored_true, "greater", "pc"),
           "ignored_gate": one_sample(ck.r_ignored_true - ck.r_ignored_mis1, "greater", "gate_ign"),
           "attended_check": one_sample(ck.r_attended_true - ck.r_attended_mis1, "greater", "gate_att"),
           "r_attended_mean": float(ck.r_attended_true.mean()), "r_ignored_mean": float(ck.r_ignored_true.mean())}
    chk["positive_control_pass"] = chk["positive_control"]["p"] < 0.05
    chk["ignored_gate_pass"] = chk["ignored_gate"]["p"] < 0.05
    res["checks"] = chk

    A = load_A(inc)
    n_used, trails, drs = {}, {}, {}
    for v in VARIANTS:
        for s in C.STREAMS:
            n, trail = mc_rule(A, v, s)
            n_used[f"{v}_{s}"], trails[f"{v}_{s}"] = n, trail
            for sub in SUB:
                drs[(v, s, sub)] = delta_r(A, v, s, n, sub)
    tab = pd.DataFrame({f"dr_{v}_{s}_{sub}": x for (v, s, sub), x in drs.items()})
    tab.index.name = "listener"
    tab.to_csv(C.TAB / "p7_A_listeners.csv")
    prim = drs[("primary", "attended", "all")] - drs[("primary", "ignored", "all")]
    a = {"n_permutations_used": n_used, "monte_carlo_trail": trails,
         "primary": one_sample(prim, "greater", "A_primary"),
         "attended": one_sample(drs[("primary", "attended", "all")], "greater", "A_att"),
         "ignored": one_sample(drs[("primary", "ignored", "all")], "two-sided", "A_ign")}
    res["A"] = a

    sync = pd.read_csv(C.TAB / "meg_attention_sync_listeners.csv").set_index("listener")
    att = drs[("primary", "attended", "all")]
    d = pd.DataFrame({"comp": part.comprehension, "sync": sync.sync, "dr": att,
                      "reject": lst.set_index("listener").reject_fraction, "r_env": ck.r_attended_true}).join(head)
    d = d.loc[[p for p in inc if p in d.index]]
    b_df = d.dropna(subset=["comp", "sync", "dr"])
    rho_s = spearman_perm(b_df.sync, b_df.comp, "B_rho_sync")
    rho_d = spearman_perm(b_df.dr, b_df.comp, "B_rho_dr")
    obs, boot = delta_rho_boot(b_df.comp, b_df.sync, b_df.dr, "p7|B|boot")
    p_dr = float((np.sum(boot <= 0) + 1) / (N_BOOT + 1))
    covs = [b_df.reject, b_df.r_env, b_df.head_x_mm, b_df.head_y_mm, b_df.head_z_mm]
    pr, pp = partial_test(b_df.sync, b_df.comp, covs, seed("B_partial"))
    step0 = json.loads((C.ROOT / "results" / "alice_listener_tests.json").read_text())
    H = pd.read_csv(C.ROOT / "results" / "alice_listeners.csv").dropna(subset=["comprehension", "isc", "delta_r_gpt2"])
    a_obs, a_boot = delta_rho_boot(H.comprehension, H.isc, H.delta_r_gpt2, "p7|alice|boot")
    alice_rho, alice_n = float(step0["delta_rho"]["rho_isc"]), int(len(H))
    w = np.array([(alice_n - 3) / 1.06, (len(b_df) - 3) / 1.06])
    z = np.array([np.arctanh(alice_rho), np.arctanh(rho_s["rho"])])
    zp, zse = float((w * z).sum() / w.sum()), float(1 / np.sqrt(w.sum()))
    se = np.array([a_boot.std(ddof=1), boot.std(ddof=1)])
    dv = np.array([a_obs, obs])
    wd = 1 / se ** 2
    dp, dse = float((wd * dv).sum() / wd.sum()), float(1 / np.sqrt(wd.sum()))
    res["B"] = {"n": int(len(b_df)), "rho_comp_sync": {**rho_s, "ci95_fisher": fisher_ci(rho_s["rho"], len(b_df))},
                "rho_comp_dr": {**rho_d, "ci95_fisher": fisher_ci(rho_d["rho"], len(b_df))},
                "delta_rho": {"observed": obs, "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                              "se_boot": float(boot.std(ddof=1)), "p_one_sided": p_dr},
                "partial_rho": {"rho": pr, "p_one_sided": pp, "covariates": ["reject_fraction", "attended_envelope_r", "head_x", "head_y", "head_z"]},
                "alice": {"n": alice_n, "rho_comp_sync": alice_rho, "delta_rho": a_obs,
                          "delta_rho_se_boot": float(a_boot.std(ddof=1))},
                "pooled": {"rho_comp_sync": float(np.tanh(zp)), "rho_ci95": [float(np.tanh(zp - 1.96 * zse)), float(np.tanh(zp + 1.96 * zse))],
                           "rho_p_one_sided": float(stats.norm.sf(zp / zse)),
                           "delta_rho": dp, "delta_rho_ci95": [dp - 1.96 * dse, dp + 1.96 * dse],
                           "delta_rho_p_one_sided": float(stats.norm.sf(dp / dse))}}
    comp = json.loads((C.TAB / "meg_attention_sync_components.json").read_text())
    res["B"]["components"] = comp
    res["B"]["sync_vs_head"] = {ax: float(stats.spearmanr(d.dropna(subset=["sync"]).sync, d.dropna(subset=["sync"])[f"head_{ax}_mm"]).correlation)
                                for ax in "xyz"}
    res["B"]["components_vs_comp"] = {f"c{c}": float(stats.spearmanr(sync.loc[b_df.index, f"sync_c{c}"], b_df.comp).correlation) for c in (1, 2, 3)}

    (C.TAB / "meg_attention_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
