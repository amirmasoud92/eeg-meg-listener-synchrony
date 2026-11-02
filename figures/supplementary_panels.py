"""Panels of the supplementary figures, drawn from the result tables in figure_data/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from common.settings import PROJECT_ROOT, load_paths
from figures import supplementary_style as S
from figures.supplementary_style import BAND, INK, INK2, INK3, LANG, LANG_L, MUSIC, MUSIC_L, RULE, SONG, SONG_L, SYNC, SYNC_L, W, axes_in
from statistical_tests.inference import dz_ci_from

ROOT = PROJECT_ROOT
OUT = PROJECT_ROOT / "figure_output"
P = load_paths()
T = PROJECT_ROOT / "figure_data"
TI, TX, P5 = T, T, T
P6 = T
DIV = LinearSegmentedColormap.from_list("div", [S.MELODY.dark, S.MELODY.base, S.MELODY.tint, "#FFFFFF", S.WORDS.tint, S.WORDS.base,
                                                 S.WORDS.dark])          # diverging: melody ochre, white midpoint, word blue
VCOL = {"speech": LANG, "song": SONG, "melody": MUSIC}
VLIGHT = {"speech": LANG_L, "song": SONG_L, "melody": MUSIC_L}
VNAME = {"speech": "spoken", "song": "sung", "melody": "hummed"}


def J(path):
    return json.loads(Path(path).read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=600)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def strip(ax, x, vals, color, s=8, seed=0, spread=(0.1, 0.34)):
    """Jittered points to the right of x; mean and 95 % CI at x."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    jit = np.random.default_rng(seed).uniform(*spread, v.size)
    ax.scatter(x + jit, v, s=s, color=color, alpha=0.75, linewidths=0, zorder=2)
    m, h = S.mean_ci(v)
    ax.plot([x, x], [m - h, m + h], color=INK, lw=1.2, zorder=3, solid_capstyle="butt")
    ax.scatter([x], [m], s=14, color="white", edgecolor=INK, linewidths=1.0, zorder=4)
    return m, h


def paired(ax, a, b, ca, cb, labels, s=9, x=(0, 1)):
    """Two conditions per unit, lines connecting units, means as bars."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    for u, v in zip(a, b):
        ax.plot(x, [u, v], color=RULE, lw=0.6, zorder=1)
    ax.scatter(np.full(a.size, x[0]), a, s=s, color=ca, linewidths=0, zorder=2)
    ax.scatter(np.full(b.size, x[1]), b, s=s, color=cb, linewidths=0, zorder=2)
    for xx, v in zip(x, (a, b)):
        ax.plot([xx - 0.22, xx + 0.22], [np.nanmean(v)] * 2, color=INK, lw=1.4, zorder=3, solid_capstyle="butt")
    ax.set_xticks(list(x), labels)
    ax.set_xlim(x[0] - 0.5, x[1] + 0.5)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)


def forest(ax, rows, xlim, band=None, xlabel="effect size (dz)", label_fs=5.8, notes_x=None):
    """rows: (label, est, lo, hi, colour, hollow, note). Drawn top to bottom."""
    n = len(rows)
    if band is not None:
        ax.axvspan(-band, band, color=BAND, lw=0, zorder=0)
    S.zero(ax, horizontal=True)
    for i, (lab, est, lo, hi, c, hollow, note) in enumerate(rows):
        y = n - 1 - i
        if est is None:                                           # group header
            ax.text(xlim[0], y, lab, fontsize=label_fs, fontweight="bold", color=c, va="center", ha="left", clip_on=False,
                    transform=ax.get_yaxis_transform(), x=-0.02)
            continue
        clo, chi = max(lo, xlim[0]), min(hi, xlim[1])
        ax.plot([clo, chi], [y, y], color=c, lw=1.0, zorder=2, solid_capstyle="butt")
        for edge, clipped, mk in ((clo, lo < xlim[0], "<"), (chi, hi > xlim[1], ">")):
            if clipped:
                ax.scatter([edge], [y], marker=mk, s=14, color=c, zorder=3, linewidths=0)
        ax.scatter([est], [y], s=15, facecolor="white" if hollow else c, edgecolor=c, linewidths=0.9, zorder=4)
        if note and notes_x is not None:
            ax.text(notes_x, y, note, fontsize=5.4, color=INK2, va="center", ha="left", transform=ax.get_yaxis_transform())
    ax.set_yticks([n - 1 - i for i, r in enumerate(rows) if r[1] is not None], [r[0] for r in rows if r[1] is not None], fontsize=label_fs)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)


def rho_ci(r, n):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def figS2():
    a1 = pd.read_csv(T / "aalborg_geometry.csv")
    prim = a1[a1.representation == "net_delta_theta"]
    conds = ["MC", "MD", "MF", "SD", "SF"]
    cname = {"MC": "cello", "MD": "Danish song", "MF": "Finnish song", "SD": "Danish speech", "SF": "Finnish speech"}
    ccol = {"MC": (MUSIC, MUSIC), "MD": (SONG, SONG), "MF": ("white", SONG), "SD": (LANG, LANG), "SF": ("white", LANG)}
    Dm = np.zeros((5, 5))
    for i, a_ in enumerate(conds):
        for j, b_ in enumerate(conds):
            if i < j:
                Dm[i, j] = Dm[j, i] = prim[f"rdm_{a_}_{b_}"].mean()
    H_ = 8.7
    fig = plt.figure(figsize=(W, H_))
    R1, R2, R3 = H_ - 0.22, 5.55, 2.15
    # (a) group RDM and its 2-D embedding
    ax = axes_in(fig, 0.85, R1 - 2.05, 1.25, 1.25)
    vmax = np.abs(Dm).max()
    ax.imshow(Dm, cmap=DIV, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    ax.set_xticks(range(5), conds, fontsize=5.6)
    ax.set_yticks(range(5), [cname[c] for c in conds], fontsize=5.6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(5):
        for j in range(5):
            if i != j:
                ax.text(j, i, f"{Dm[i, j]:.2f}".replace("-", "−"), ha="center", va="center", fontsize=4.8,
                        color="white" if abs(Dm[i, j]) > 0.55 * vmax else INK)
    ax.set_title("crossnobis distance", fontsize=6, color=INK2, pad=3)
    bx = axes_in(fig, 2.55, R1 - 2.05, 1.25, 1.25)
    Jm = np.eye(5) - 1 / 5
    Bm = -0.5 * Jm @ (Dm ** 2) @ Jm
    w, V = np.linalg.eigh(Bm)
    idx = np.argsort(w)[::-1][:2]
    Y = V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))
    for k, c in enumerate(conds):
        fc, ec = ccol[c]
        bx.scatter(Y[k, 0], Y[k, 1], s=42, facecolor=fc, edgecolor=ec, linewidths=1.1, zorder=3)
        off, va = {"SF": ((6, 0), "center"), "MF": ((6, -5), "top")}.get(c, ((4, 4), "baseline"))   # clear of the neighbours
        bx.annotate(cname[c], (Y[k, 0], Y[k, 1]), xytext=off, textcoords="offset points", fontsize=5.6, color=ec, va=va)
    lim = np.abs(Y).max() * 1.6
    bx.set_xlim(-lim, lim)
    bx.set_ylim(-lim, lim)
    bx.set_xticks([])
    bx.set_yticks([])
    bx.set_xlabel("MDS 1", labelpad=1)
    bx.set_ylabel("MDS 2", labelpad=1)
    bx.set_title("open: Finnish (not understood)", fontsize=6, color=INK2, pad=3)
    S.fig_heading(fig, 0.3, R1, "a", "Network-state geometry", "Aalborg, 18 listeners; group mean of the cross-validated distances (δ+θ network features)")
    # (b) the language-capture index per listener, analysis variants and acoustic controls
    ax = axes_in(fig, 4.55, R1 - 2.05, 1.75, 1.35)
    variants = [("net_delta_theta", "primary"), ("net_delta", "δ"), ("net_theta", "θ"), ("net_delta_theta_no_aperiodic", "no\naperiodic"), ("full_delta_theta", "full\nδ+θ")]
    S.zero(ax)
    for k, (rep, nm) in enumerate(variants):
        v = a1[a1.representation == rep].lci
        strip(ax, k, v, SYNC, seed=k)
        dz = v.mean() / v.std(ddof=1)
        ax.text(k + 0.2, 1.02, f"dz {S.num(dz)}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.4, color=INK2)
    ax.set_xticks([k + 0.2 for k in range(5)], [v[1] for v in variants], fontsize=5.6)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.4, 4.8)
    ax.set_ylabel("language-capture index")
    cx = axes_in(fig, 6.55, R1 - 2.05, 0.5, 1.35)
    ac = J(T / "stimulus_benchmark.json")
    S.zero(cx)
    strip(cx, 0, ac["lci"], INK3, seed=7)
    cx.text(0.2, 1.02, f"dz {S.num(ac['summary']['dz'])}", transform=cx.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.4, color=INK2)
    cx.set_xticks([0.2], ["stimulus\nenvelopes"], fontsize=5.6)
    cx.tick_params(axis="x", length=0)
    cx.spines["bottom"].set_visible(False)
    cx.set_xlim(-0.4, 0.8)
    S.fig_heading(fig, 4.15, R1, "b", "The index per listener", "analysis variants; right: the same index from the audio alone")
    # (c) the 20 robustness variants
    rb = pd.read_csv(T / "aalborg_network_variants.csv")
    tcol = {"main": SYNC, "variant": INK2, "robustness": INK3}
    rows = [(r.analysis.replace("network", "net.").replace("delta theta", "δ+θ").replace("delta", "δ").replace("theta", "θ").replace("alpha", "α").replace("beta", "β"),
             r.dz, r.dz_lo, r.dz_hi, tcol[r.group], "envelope" in r.analysis, None) for r in rb.itertuples()]
    ax = axes_in(fig, 2.55, R2 - 2.85, 1.85, 2.6)
    forest(ax, rows, (-1.3, 2.5), band=0.611, xlabel="dz with 95 % CI; band: equivalence bounds")
    S.fig_heading(fig, 0.3, R2, "c", "Analysis variants", "graphite primary, grey secondary, light robustness; open: stimulus benchmark")
    # (d) specification curves
    inf = {"aalborg": pd.read_csv(TX / "specifications_aalborg.csv"), "etard": pd.read_csv(TX / "specifications_etard.csv")}
    panels = [("aalborg", "lci", "H1 language-capture index"), ("aalborg", "alignment", "H1b axis alignment"),
              ("aalborg", "auc", "speech → song cross-decoding"), ("etard", "delta", "HR1 Etard Δ")]
    for k, (ds, st, ttl) in enumerate(panels):
        g = inf[ds][inf[ds].stat == st].sort_values("dz").reset_index(drop=True)
        ax = axes_in(fig, 5.0, R2 - 0.85 - k * 0.68, 2.0, 0.5)
        col = np.where(g.p_fwe < 0.05, INK, np.where(g.p_unc < 0.05, S.GLOBAL.mid, SYNC_L))
        ax.scatter(np.arange(len(g)), g.dz, s=3, c=col, linewidths=0)
        S.zero(ax)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_xlim(-3, len(g) + 2)
        ax.set_ylabel("dz", labelpad=1)
        nf, nu = int((g.p_fwe < 0.05).sum()), int((g.p_unc < 0.05).sum())
        ax.set_title(f"{ttl}: {nf}/{len(g)} FWE, {nu} uncorrected", fontsize=5.6, color=INK2, pad=2, loc="left")
    ax.set_xlabel("specifications sorted by dz", labelpad=2)
    S.fig_heading(fig, 4.6, R2, "d", "Multiverse", "black: p$_{FWE}$ < .05; grey: p < .05 uncorrected")
    # (e) the replication HR1 (Etard)
    r5 = pd.read_csv(T / "etard_contrast_listeners.csv")
    e5 = J(T / "etard_contrast.json")
    s5 = J(T / "etard_contrast_sensor.json")["primary"]
    x5 = J(T / "etard_contrast_all_listeners.json")["primary"]
    ax = axes_in(fig, 0.75, 0.45, 1.3, 1.3)
    paired(ax, r5.d_en, r5.d_nl, LANG, INK3, ["English\n(understood)", "Dutch\n(not)"])
    S.zero(ax)
    ax.set_ylabel("within-session distance\nlow → high babble")
    bx = axes_in(fig, 2.55, 0.45, 0.8, 1.3)
    S.zero(bx)
    strip(bx, 0, r5.delta, SYNC, seed=3)
    bx.set_xticks([0.2], ["Δ = d$_{EN}$ − d$_{NL}$"], fontsize=5.8)
    bx.tick_params(axis="x", length=0)
    bx.spines["bottom"].set_visible(False)
    bx.set_xlim(-0.4, 0.8)
    bx.text(0.2, 1.02, f"dz {S.num(e5['primary']['dz'])}, BF$_{{+0}}$ {e5['bf10_plus']:.2f}", transform=bx.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.6, color=INK2)
    rows = [("eLORETA (primary)", e5["primary"]), ("sensitivity-scaled", e5["sensitivity_scaled"]), ("acoustic-adjusted", e5["acoustic_adjusted"]),
            ("graded contrast", e5["graded"]), ("sensor space", s5), ("YH14 included (n = 12)", x5)]
    rows = [(nm, d["dz"], *d["dz_ci95"], SYNC if k == 0 else INK2, False, None) for k, (nm, d) in enumerate(rows)]
    cx = axes_in(fig, 5.0, 0.45, 2.0, 1.3)
    forest(cx, rows, (-1.6, 0.8), xlabel="Δ effect size (dz, 95 % CI)")
    S.fig_heading(fig, 0.3, R3, "e", "Replication on independent data (HR1)", "Etard & Reichenbach, 11 listeners: the representation detects babble within each language, but Δ is negative")
    save(fig, "FigS2")


def figS3():
    d = pd.read_csv(TI / "speech_in_noise_passages.csv")
    res = J(TI / "speech_in_noise.json")
    fig = plt.figure(figsize=(W, 5.2))
    R1, R2 = 4.98, 2.4
    cols = {"a": ("audio only", INK3), "v": ("video only", SYNC_L), "av": ("audiovisual", SYNC)}
    ax = axes_in(fig, 0.75, R1 - 2.05, 2.5, 1.45)
    for _, r in d.iterrows():
        ax.plot([0, 1, 2], [r.isc_a, r.isc_v, r.isc_av], color=RULE, lw=0.6, zorder=1)
    for x, (c, (nm, col)) in enumerate(cols.items()):
        ax.scatter(np.full(len(d), x), d[f"isc_{c}"], s=10, color=col, edgecolor=SYNC if c == "v" else "none", linewidths=0.5, zorder=2)
        ax.plot([x - 0.22, x + 0.22], [d[f"isc_{c}"].mean()] * 2, color=INK, lw=1.4, zorder=3, solid_capstyle="butt")
    ax.set_xticks([0, 1, 2], [v[0] for v in cols.values()])
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("δ synchrony (speech-defined space)")
    g = res["sec1_isc_gain_av_minus_a"]
    S.fig_heading(fig, 0.3, R1, "a", "The video alone synchronises listeners", f"15 passages; audiovisual − audio dz {g['dz']:.2f}, {g['n_positive']}/15 passages")
    ax = axes_in(fig, 4.55, R1 - 2.05, 2.3, 1.45)
    pred, obs = (d.S_a + d.S_v + 2 * d.C_av) / d.T_a, d.S_av / d.T_a
    lim = [0, max(pred.max(), obs.max()) * 1.12]
    ax.plot(lim, lim, color=INK3, lw=0.6, ls=(0, (2, 2)))
    ax.scatter(pred, obs, s=14, color=SYNC, linewidths=0, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("additive prediction (S$_A$ + S$_V$ + 2C)")
    ax.set_ylabel("observed audiovisual\nshared variance")
    p = res["primary_items_D"]
    S.fig_heading(fig, 4.0, R1, "b", "No superadditivity", f"superadditivity: dz {S.num(p['dz'])} [{S.num(p['dz_ci95'][0])}, {S.num(p['dz_ci95'][1])}], {p['n_positive']}/15 passages above the line")
    ra, rav = pd.read_csv(TI / "speech_in_noise_ratings_audio.csv"), pd.read_csv(TI / "speech_in_noise_ratings_audiovisual.csv")
    ma, mav = ra.groupby("passage").rating.mean() * 10, rav.groupby("passage").rating.mean() * 10
    ax = axes_in(fig, 0.75, 0.45, 1.6, 1.4)
    paired(ax, ma.reindex(mav.index), mav, INK3, SYNC, ["audio only", "audiovisual"])
    ax.set_ylabel("words understood (%, passage mean)")
    S.fig_heading(fig, 0.3, R2, "c", "The manipulation worked", f"understanding {ma.mean():.0f} → {mav.mean():.0f} %")
    keys = [("sec3_graded_a_rows1-21", "audio, rows 1–21"), ("sec3_graded_a_rows2-22", "audio, rows 2–22"),
            ("sec3_graded_av_rows1-21", "audiovisual, rows 1–21"), ("sec3_graded_av_rows2-22", "audiovisual, rows 2–22")]
    rows = [(lab, res[k]["dz"], *res[k]["dz_ci95"], INK2, False, None) for k, lab in keys]
    ax = axes_in(fig, 4.55, 0.45, 2.3, 1.4)
    forest(ax, rows, (-1.1, 0.9), xlabel="dz of the within-listener rating–synchrony correlation")
    S.fig_heading(fig, 3.2, R2, "d", "Graded understanding is uninterpretable", "the sign depends on the undocumented rating-row mapping (22 rows, 21 listeners)")
    save(fig, "FigS3")


def figS4():
    E = pd.read_csv(TI / "cocktail_party_isc.csv")
    e = J(TI / "cocktail_party.json")
    dec = pd.read_csv(TX / "attention_decoding.csv").sort_values("accuracy")
    lm = E.groupby("listener")[["same", "other"]].mean()
    fig = plt.figure(figsize=(W, 2.75))
    R1 = 2.53
    ax = axes_in(fig, 0.75, 0.45, 1.45, 1.5)
    paired(ax, lm.same, lm.other, SYNC, INK3, ["same story", "other story"])
    ax.set_ylabel("δ synchrony (speech space)")
    e1 = e["runs_same_minus_other"]
    S.fig_heading(fig, 0.3, R1, "a", "No credible attention effect on synchrony", f"listener means over runs; same − other story over 30 runs: dz {S.num(e1['dz'])}")
    ax = axes_in(fig, 3.2, 0.45, 1.25, 1.5)
    ax.axhspan(0, 50, color=BAND, lw=0, zorder=0)
    ax.scatter(np.arange(len(dec)), dec.accuracy * 100, s=11, color=SYNC, linewidths=0, zorder=2)
    ax.axhline(dec.accuracy.mean() * 100, color=INK, lw=0.7)
    ax.text(0.5, dec.accuracy.mean() * 100 + 1.5, f"mean {dec.accuracy.mean() * 100:.0f} %", fontsize=5.6, color=INK, va="bottom")
    ax.text(0.5, 47.5, "chance", fontsize=5.4, color=INK2, ha="left", va="top")
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(30, 104)
    ax.set_xlabel("listeners, sorted")
    ax.set_ylabel("attended story decoded (% runs)")
    S.fig_heading(fig, 2.8, R1, "b", "The same EEG carries attention", "backward decoder, per listener")
    rows = [("same − other story", e1, "30 runs"), ("attended comprehension", e["graded_attended_score_vs_isc_same"], "33 listeners"),
            ("ignored comprehension", e["graded_attended_score_vs_isc_other"], "33 listeners")]
    rows = [(f"{nm}\n{u}", d["dz"], *d["dz_ci95"], SYNC, False, None) for nm, d, u in rows]
    ax = axes_in(fig, 5.75, 0.45, 1.2, 1.5)
    forest(ax, rows, (-0.9, 0.9), band=0.5, xlabel="dz with 95 % CI")
    S.fig_heading(fig, 5.1, R1, "c", "Tests within listeners", f"behaviour: {e['comprehension_means']['attended'] * 100:.0f} % correct attended vs {e['comprehension_means']['unattended'] * 100:.0f} % ignored")
    save(fig, "FigS4")


def figS5():
    from scipy import stats
    tr = pd.read_csv(TI / "engagement_trials.csv")
    tr["isc_c"] = tr.isc - tr.groupby("subject").isc.transform("mean") - tr.groupby("condition").isc.transform("mean") + tr.isc.mean()
    per = tr.groupby("subject").apply(lambda g: stats.pearsonr(g.eng_c, g.isc_c)[0] if g.eng_c.std() > 0 else np.nan, include_groups=False)
    al = pd.read_csv(TI / "alice_alpha_per_segment.csv", index_col=0)
    isc = pd.read_csv(TX / "alice_isc_per_segment.csv").pivot(index="listener", columns="segment", values="isc")
    isc.columns = [f"seg{c}" for c in isc.columns]
    rs = []
    for lst in al.index.intersection(isc.index):
        a, b = al.loc[lst], isc.loc[lst].reindex(al.columns)
        ok = a.notna() & b.notna()
        if ok.sum() >= 6:
            rs.append(stats.pearsonr(a[ok], b[ok])[0])
    I = J(TI / "engagement_alpha.json")
    fig = plt.figure(figsize=(W, 2.6))
    R1 = 2.38
    for k, (vals, nm, sub, x0) in enumerate(((per.dropna(), "Rated engagement", f"Aalborg, 18 listeners; per-trial ratings, centred; dz {S.num(I['aalborg']['all']['dz'])}", 0.85),
                                            (np.array(rs), "Posterior α state", f"Alice, {len(rs)} listeners; α per one-minute segment; dz {S.num(I['alice']['within_alpha_isc_negative']['dz'])}", 4.15))):
        ax = axes_in(fig, x0, 0.42, 1.6, 1.45)
        S.zero(ax)
        strip(ax, 0, vals, SYNC, seed=k)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_xlim(-0.4, 0.8)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("within-listener r with synchrony")
        S.fig_heading(fig, x0 - 0.45, R1, "ab"[k], nm, sub)
    save(fig, "FigS5")
