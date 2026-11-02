"""Main Figures 1-6."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as ss
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Rectangle

from common.settings import PROJECT_ROOT
from figures import design as V, shared_panels as PM
from figures.design import BAND, DZ, INK, INK2, INK3, INKR, MELODY, PAPER, RULE, SAND, SUNG, WORDS

ROOT = PROJECT_ROOT
OUT = PROJECT_ROOT / "figure_output"
J, P = PM.J, PM.P
TI, TX, P5, P6, T7, T78, T8, T9 = PM.TI, PM.TX, PM.P5, PM.P6, PM.T7, PM.T78, PM.T8, PM.T9
SOURCES = [("The signal", "Figs. 2, 5e", "Only stimulus-driven activity\ncan be shared.",
            "Prediction: synchrony falls when\nspeech is reversed or its acoustic\ndrive is removed."),
           ("The listener's state", "Fig. 3", "Understanding, attention or\nengagement at the moment.",
            "Prediction: synchrony changes\nwith the state when the signal\nis held fixed."),
           ("The listener", "Figs. 4, 5", "A stable property of each\nlistener's response.",
            "Prediction: synchrony is stable\nacross days and across kinds\nof sound.")]
CONTRASTS = [("The signal", "Figs. 2, 5e", ["intact vs time-reversed", "sentences vs word lists"]),
             ("The listener's state", "Fig. 3", ["seeing the talker", "attention", "run-wise comprehension", "engagement, α state"]),
             ("The listener", "Figs. 4, 5", ["comprehension", "stability (segments, days)", "across kinds of sound"]),
             ("Song", "Fig. 6", ["understood vs foreign", "sung vs hummed"]),
             ("Network states", "Suppl. Note 1", ["understood vs foreign"]),
             ("Word-level reference", "Figs. 4d, 6c", ["word level"])]
DATASETS = [("Aalborg songs and speech", "EEG", 18, {5: "f", 7: "f", 8: "f", 9: "f", 11: "o"}),
            ("Etard and Reichenbach", "EEG", 11, {11: "o"}),
            ("Broderick: natural speech", "EEG", 19, {0: "f"}),
            ("Broderick: time-reversed", "EEG", 10, {0: "f"}),
            ("Broderick: speech in noise", "EEG", 21, {2: "f"}),
            ("Broderick: cocktail party", "EEG", 33, {3: "f", 4: "f"}),
            ("Alice", "EEG", 48, {5: "f", 6: "f", 7: "f", 12: "s"}),
            ("CantiSong", "EEG", 20, {8: "f", 10: "f", 12: "s"}),
            ("Two audiobooks, one attended", "MEG", 58, {6: "f", 12: "s"}),
            ("MEG-MASC, two sessions", "MEG", 27, {1: "f", 7: "f"})]


def _illustration(name):
    with np.load(PROJECT_ROOT / "figure_data" / "illustrations.npz") as z:
        return {k.split("|", 1)[1]: z[k] for k in z.files if k.startswith(name + "|")}


def _song01_component():
    """CantiSong song 01, sung: the first correlated component (delta band), fitted on the other sung songs; its forward
    model (scalp pattern), the component in every listener and each listener's ISC with the mean of the others (display
    only). Computed by figures/illustration_data.py."""
    d = _illustration("song01")
    return d["chans"].tolist(), d["A"], d["Y"], d["isc"], d["t"]


def fig1_measure(fig, y0):
    chans, A, Y, isc, t = _song01_component()
    w0, w1 = 6.0, 16.0
    m = (t >= w0) & (t <= w1)
    # scalp pattern of the component
    a = V.ax(fig, 0.30, y0 + 0.28, 1.22, 1.22)
    V.topomap(a, V.eeg_xy(chans), A / np.abs(A).max(), cmap=V.MAP_FIELD, norm=Normalize(-1, 1), contours=5, sensor_size=1.1)
    V.text(fig, 0.91, y0 + 0.16, "scalp pattern", ha="center", color=INK2)
    V.text(fig, 0.91, y0 + 1.70, "component 1 (δ band)", ha="center", color=INK2)
    V.text(fig, 0.91, y0 + 1.86, "sign arbitrary", ha="center", color=INK3)
    # its time course in five listeners, and the mean over all
    c = V.canvas(fig, 1.62, y0, 3.86, 2.10)
    X0, X1 = 0.46, 3.72
    tx = lambda tt: X0 + (tt - w0) / (w1 - w0) * (X1 - X0)
    show = [0, 3, 7, 11, 15]
    for k, i in enumerate(show):
        yc = 0.22 + 0.205 * k
        c.plot(tx(t[m]), yc - 0.032 * np.clip(Y[i, m], -3.2, 3.2), color=INKR.mid, lw=0.55)
        c.text(X0 - 0.08, yc, f"{k + 1}", ha="right", va="center", color=INK3)
    c.text(0.10, 0.63, "listener", rotation=90, ha="center", va="center", color=INK3)
    ym = 1.50
    mt = Y[:, m].mean(0)
    c.plot(tx(t[m]), ym - 0.12 * mt / mt.std(), color=INK, lw=1.0)
    c.text(X0 - 0.08, ym, "mean", ha="right", va="center", color=INK)
    c.plot([tx(w1 - 2), tx(w1)], [1.92] * 2, color=INK, lw=1.0, solid_capstyle="butt")
    c.text(tx(w1 - 1), 2.06, "2 s", ha="center", va="baseline", color=INK2)
    c.text(X0, 2.06, "CantiSong, song 01 (sung)", ha="left", va="baseline", color=INK3)
    V.arrow(V.canvas(fig, 1.50, y0, 0.24, 2.0), (0.01, 0.89), (0.22, 0.89), color=INK3, lw=0.8)
    # ISC: each listener's component correlated with the mean of the others
    br = V.canvas(fig, 5.47, y0, 0.50, 2.10)
    br.plot([0.03, 0.09, 0.09, 0.03], [0.14, 0.14, 1.66, 1.66], color=INK3, lw=0.7)
    V.arrow(br, (0.09, 0.90), (0.40, 0.90), color=INK3, lw=0.8)
    s_ = V.ax(fig, 6.32, y0 + 0.28, 0.40, 1.38)
    V.clean(s_, bottom=False)
    s_.set_xlim(-0.6, 0.6)
    hi = float(np.ceil(isc.max() * 10) / 10)
    s_.set_ylim(0, hi)
    s_.set_yticks([0, hi / 2, hi], ["0", V.num(hi / 2, ".2g"), V.num(hi, ".2g")])
    xs = V.swarm(s_, 0.0, isc, INKR.light, size_pt=4.2, edge=INKR.mid, max_pt=12)
    for i in show:
        s_.scatter([xs[i]], [isc[i]], s=18, color=INK, edgecolors=PAPER, linewidths=0.5, zorder=5)
    s_.set_ylabel("ISC", labelpad=2)
    V.text(fig, 6.40, y0 + 0.16, "ISC per listener", ha="center", color=INK2)
    V.text(fig, 6.40, y0 + 1.86, f"{len(isc)} listeners", ha="center", color=INK3)


def _head(c, x, y, r=0.12, fc=PAPER, ec=INK, lw=0.8, z=2):
    """A head seen from above, nose up (inch canvas, y down), as on the scalp maps."""
    from matplotlib.patches import Circle
    c.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z))
    c.add_patch(Polygon([(x - 0.13 * r, y - 0.99 * r), (x, y - 1.22 * r), (x + 0.13 * r, y - 0.99 * r)], closed=False, fill=False,
                        ec=ec, lw=lw, zorder=z, joinstyle="round"))
    for s in (-1, 1):
        c.add_patch(Polygon([(x + s * 0.99 * r, y - 0.15 * r), (x + s * 1.12 * r, y - 0.10 * r), (x + s * 1.12 * r, y + 0.16 * r),
                             (x + s * 0.99 * r, y + 0.20 * r)], closed=False, fill=False, ec=ec, lw=lw, zorder=z, joinstyle="round"))


def _wave(c, x, y, w, color, rev=False, amp=0.10):
    tt = np.linspace(0, 1, 220)
    env = np.exp(-((tt - 0.28) / 0.12) ** 2) + 0.75 * np.exp(-((tt - 0.70) / 0.10) ** 2) + 0.15
    sig = env * np.sin(2 * np.pi * 10 * tt)
    c.plot(x + w * tt, y + amp * (sig[::-1] if rev else sig), color=color, lw=0.8)


def _icon_signal(c, x, y):
    _wave(c, x, y, 0.62, INK)
    _wave(c, x + 0.86, y, 0.62, INK3, rev=True)
    c.text(x + 0.31, y + 0.30, "intact", ha="center", va="center", color=INK)
    c.text(x + 1.17, y + 0.30, "reversed", ha="center", va="center", color=INK3)


def _icon_state(c, x, y):
    _wave(c, x, y, 0.50, INK, amp=0.08)
    c.text(x + 0.25, y + 0.30, "same sound", ha="center", va="center", color=INK2)
    V.arrow(c, (x + 0.56, y), (x + 0.76, y - 0.12), color=INK3, lw=0.7, ms=5)
    V.arrow(c, (x + 0.56, y), (x + 0.76, y + 0.12), color=INK3, lw=0.7, ms=5)
    _head(c, x + 0.92, y - 0.13, r=0.075, fc=INKR.base, ec=INKR.base)
    _head(c, x + 0.92, y + 0.15, r=0.075, fc=PAPER, ec=INK3)
    c.text(x + 1.06, y - 0.13, "understood", ha="left", va="center", color=INK)
    c.text(x + 1.06, y + 0.15, "not understood", ha="left", va="center", color=INK3)


def _icon_listener(c, x, y):
    for k, lab in enumerate(("day 1", "day 2")):
        cx = x + 0.16 + 0.78 * k
        _head(c, cx, y, r=0.11, fc=INKR.light, ec=INK)
        c.add_patch(Rectangle((cx + 0.20, y - 0.11), 0.06, 0.22, fc=PAPER, ec=INK3, lw=0.6, zorder=2))
        c.add_patch(Rectangle((cx + 0.20, y - 0.11 + 0.22 * 0.35), 0.06, 0.22 * 0.65, fc=INK, ec="none", zorder=3))
        c.text(cx + 0.06, y + 0.30, lab, ha="center", va="center", color=INK2)
    c.text(x + 0.62, y, "=", ha="center", va="center", color=INK, fontsize=10)
    c.text(x + 1.24, y, "ISC", ha="left", va="center", color=INK2)


def fig1_sources(fig, y0):
    c = V.canvas(fig, 0, y0, V.W, 1.86)
    xs = [0.30, 2.66, 5.02]
    icons = [_icon_signal, _icon_state, _icon_listener]
    for (name, ref, what, pred), x, icon in zip(SOURCES, xs, icons):
        c.plot([x, x + 2.06], [0.02, 0.02], color=INK, lw=1.0, solid_capstyle="butt")
        c.text(x, 0.22, name, fontsize=V.FS["head"], fontweight="bold", color=INK, va="baseline")
        c.text(x + 2.06, 0.22, ref, color=INK2, ha="right", va="baseline")
        icon(c, x + 0.10, 0.55)
        c.text(x, 1.04, what, color=INK, va="top", linespacing=1.2)
        c.text(x, 1.40, pred, color=INK2, va="top", linespacing=1.2)
    return c


def fig1_datasets(fig, y0):
    """The datasets linked to the contrasts each isolates, grouped by the source of variation they test."""
    contrasts = [(g, r, cols, WORDS.dark if g == "Word-level reference" else None) for g, r, cols in CONTRASTS]
    return V.link_diagram(fig, y0, contrasts, DATASETS, ref_color=WORDS.base,
                          key=[("f", "tested in the main text"), ("o", "Supplementary Note 1"), ("s", "word-level reference")])


def figure1():
    fig = V.figure(8.08)
    V.panel(fig, 0.30, 0.20, "a", "What synchrony measures")
    fig1_measure(fig, 0.34)
    V.panel(fig, 0.30, 2.62, "b", "Three possible sources of a shared response")
    fig1_sources(fig, 2.78)
    V.panel(fig, 0.30, 4.80, "c", "Ten public datasets and the contrasts they isolate")
    fig1_datasets(fig, 4.86)
    V.save(fig, OUT, "Figure1")


def fig2_cortex(fig, y0):
    """(a) exploratory: regional delta ISC, Danish (understood) minus Finnish (foreign), speech and songs (Aalborg)."""
    c = pd.read_csv(TX / "regional_synchrony.csv")
    norm = Normalize(0, 1.6)
    views = (("left", "left lateral"), ("left_medial", "left medial"), ("right_medial", "right medial"), ("right", "right lateral"))
    x0, bw, bh, dx = 1.30, 1.30, 0.86, 1.33
    rows = (("SD-SF", "speech", WORDS.dark, y0 + 0.20), ("MD-MF", "songs", SUNG.dark, y0 + 1.30))
    for con, nm, col, y in rows:
        g = c[c.contrast == con].set_index("region")
        sig = set(g.index[g.q_bh < 0.05])
        for j, (v, vl) in enumerate(views):
            a = V.ax(fig, x0 + j * dx, y, bw, bh)
            V.brain(a, g.dz.to_dict(), v, V.MAP_SYNC, norm, faded=set(g.index) - sig, outline="#FFFFFF", outline_lw=0.25, fade=0.78)
            if con == "SD-SF":
                V.text(fig, x0 + j * dx + bw / 2, y - 0.04, vl, ha="center", color=INK3)
            if con == "SD-SF" and v == "right":
                cx, cy = V.region_xy("TTG_rh", "right")
                a.annotate("auditory cortex", xy=(cx, cy), xytext=(0.62, -0.02), textcoords="axes fraction", color=INK, ha="center",
                           va="top", arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=1, shrinkB=0),
                           path_effects=V.halo(2.4))
        V.text(fig, 0.30, y + 0.36, nm, fontweight="bold", color=col, fontsize=V.FS["head"])
        V.text(fig, 0.30, y + 0.53, f"{len(sig)} of {len(g)} regions", color=INK)
        if con == "SD-SF":
            V.text(fig, 0.30, y + 0.69, "not STG, IFG", color=INK3)
    V.colorbar(fig, 6.70, y0 + 0.45, 0.07, 1.30, V.MAP_SYNC, norm, [0, 0.5, 1, 1.5], ["0", "0.5", "1", "1.5"], label=DZ, horizontal=False)
    V.text(fig, 1.30, y0 + 2.34, "Understood − foreign δ synchrony per region; faded, q ≥ 0.05; exploratory, template head model",
           color=INK3)


def fig2_runs(fig, x, y_top, s=1.95):
    """(b) every audiobook run: forward against time-reversed synchrony, for all activity and beyond the acoustics."""
    Ar = pd.read_csv(TI / "intact_reversed_runs.csv").query("fit == 'pooled'").sort_values("run").reset_index(drop=True)
    Fr = pd.read_csv(TI / "acoustic_removal_runs_L1_acoustic.csv").query("fit == 'pooled'").sort_values("run").reset_index(drop=True)
    A = J(TI / "intact_reversed.json")["primary"]["pooled"]["runs_forward_minus_reversed"]
    F = J(TI / "acoustic_removal.json")["primary"]
    a = V.ax(fig, x, y_top, s, s)
    lim = (0.0, 0.27)
    a.fill_between(lim, lim, [lim[1]] * 2, color="#F6F6F4", lw=0, zorder=0)
    a.plot(lim, lim, color=INK3, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for i in range(len(Ar)):
        a.plot([Ar.isc_reversed[i], Fr.isc_reversed[i]], [Ar.isc_forward[i], Fr.isc_forward[i]], color=INKR.light, lw=0.6, zorder=1)
    a.scatter(Ar.isc_reversed, Ar.isc_forward, s=22, color=INK, edgecolors=PAPER, linewidths=0.5, zorder=3)
    a.scatter(Fr.isc_reversed, Fr.isc_forward, s=20, facecolor=PAPER, edgecolors=INK, linewidths=0.9, zorder=3)
    a.set_xlim(*lim)
    a.set_ylim(*lim)
    a.set_xticks([0, 0.1, 0.2], ["0", "0.1", "0.2"])
    a.set_yticks([0, 0.1, 0.2], ["0", "0.1", "0.2"])
    a.set_xlabel("time-reversed speech (ISC)")
    a.set_ylabel("intact speech (ISC)")
    a.text(0.262, 0.252, "intact = reversed", rotation=45, rotation_mode="anchor", ha="right", va="top", color=INK3,
           transform=a.transData)
    lx = x + s + 0.10
    for k, (mk, lab, st) in enumerate((("o", "all activity", A), ("o_", "beyond the acoustics", F))):
        yy = y_top + 0.10 + k * 0.62
        cnv = V.canvas(fig, lx, yy - 0.10, 0.20, 0.20)
        cnv.plot(0.08, 0.10, "o", ms=5.0, mfc=INK if mk == "o" else PAPER, mec=INK, mew=0.9)
        V.text(fig, lx + 0.20, yy + 0.04, lab, color=INK)
        V.text(fig, lx + 0.20, yy + 0.21, f"{st['n_positive']} of {st['n']} runs above", color=INK2)
        V.text(fig, lx + 0.20, yy + 0.38, f"{DZ} = {V.num(st['dz'])}", color=INK2)
    return a


def fig2_removal(fig, x, y_top, w=2.30, h=1.95):
    """(c) the forward - reversed advantage after cumulative removal of the activity predictable from the acoustics,
    phonetic features and word onsets, with the mismatched-phonetic control; share of the original advantage."""
    F = J(TI / "acoustic_removal.json")
    frac = F["fraction_of_L0_advantage_remaining"]["pooled"]
    lv = ["L0_none", "L1_acoustic", "L2_phonetic", "L3_word"]
    runs = {k: pd.read_csv(TI / f"acoustic_removal_runs_{k}.csv").query("fit == 'pooled'").sort_values("run") for k in lv + ["L2_mismatched"]}
    a = V.ax(fig, x, y_top, w, h)
    xs = np.array([0, 1, 2.2, 3.2])
    mh = [V.mean_ci(runs[k].isc_forward.to_numpy() - runs[k].isc_reversed.to_numpy()) for k in lv]
    m = np.array([u for u, _ in mh])
    hw = np.array([v for _, v in mh])
    a.fill_between(xs, m - hw, m + hw, color=INKR.tint, lw=0, zorder=0, step=None)
    a.plot(xs, m, color=INK, lw=1.1, zorder=2)
    for xx, mm, hh, k in zip(xs, m, hw, lv):
        V.estimate(a, xx, mm, (mm - hh, mm + hh), INK, horizontal=False, ms=5.4)
        a.text(xx, mm + hh + 0.005, f"{frac[k] * 100:.0f}%", ha="center", va="bottom", color=INK)
    d = runs["L2_mismatched"].isc_forward.to_numpy() - runs["L2_mismatched"].isc_reversed.to_numpy()
    mm, hh = V.mean_ci(d)
    V.estimate(a, 1.6, mm, (mm - hh, mm + hh), INK3, horizontal=False, ms=5.0, filled=False, marker="D")
    a.text(1.6, mm - hh - 0.006, "control", ha="center", va="top", color=INK3)
    a.set_xticks(xs, ["nothing", "acoustics", "+ phonetic", "+ word\nonsets"])
    a.tick_params(axis="x", length=0, pad=3)
    a.spines["bottom"].set_visible(False)
    a.set_xlim(-0.45, 3.6)
    a.set_ylim(0, 0.105)
    a.set_yticks([0, 0.05, 0.1], ["0", "0.05", "0.1"])
    a.set_ylabel("intact − reversed ISC")
    a.set_xlabel("activity removed before synchrony")
    return a


def figure2():
    fig = V.figure(5.86)
    V.panel(fig, 0.30, 0.20, "a", "Where understood and foreign speech differ in synchrony")
    fig2_cortex(fig, 0.36)
    V.panel(fig, 0.30, 3.14, "b", "Intact speech, run by run")
    fig2_runs(fig, 0.78, 3.38)
    V.panel(fig, 4.18, 3.14, "c", "What survives removing the acoustic drive")
    fig2_removal(fig, 4.62, 3.38)
    V.save(fig, OUT, "Figure2")


def fig3_profile(fig, x_lab, x_ax, y_top, w=1.95, xb=None, wb=0.62):
    from statistical_tests.inference import dz_ci_from
    A = J(TI / "intact_reversed.json")["primary"]["pooled"]["runs_forward_minus_reversed"]
    F = J(TI / "acoustic_removal.json")["primary"]
    ab = J(P5 / "evidence_absence_icc.json")["absence"]
    i3 = J(P6 / "cocktail_party_residuals.json")["residual_refit"]["same_minus_other_runs"]
    bf3 = 1 / J(P6 / "multiplicity.json")["bayes_factors_bf10"]["3_two_sided"]
    rows = [("group", "Signal changes"),
            ("row", "intact vs time-reversed", "20 runs", A["dz"], A["dz_ci95"], A["n"], None),
            ("row", "beyond the acoustic drive", "20 runs", F["dz"], F["dz_ci95"], F["n"], None),
            ("gap",),
            ("group", "State changes, signal fixed"),
            ("row", "seeing the talker", "15 passages", ab["D"]["dz"], None, ab["D"]["n"], ab["D"]["bf01_two_sided"]),
            ("row", "attending one story", "30 runs", ab["E1"]["dz"], None, ab["E1"]["n"], ab["E1"]["bf01_two_sided"]),
            ("row", "  envelope responses removed", "30 runs", i3["dz"], i3["dz_ci95"], i3["n"], bf3),
            ("row", "run-wise comprehension", "33 listeners", ab["E2"]["dz"], None, ab["E2"]["n"], ab["E2"]["bf01_two_sided"]),
            ("row", "rated engagement", "18 listeners", ab["I1"]["dz"], None, ab["I1"]["n"], ab["I1"]["bf01_two_sided"]),
            ("row", "posterior α state", "48 listeners", ab["I3"]["dz"], None, ab["I3"]["n"], ab["I3"]["bf01_two_sided"]),
            ("pooled", "pooled, one test per dataset", ab["pooled_D_E2_I1_I3"])]
    dy = 0.205
    ys, y = [], 0.0
    for r in rows:
        ys.append(y)
        y += dy * (0.5 if r[0] == "gap" else 1.0)
    h = y + 0.05
    a = V.ax(fig, x_ax, y_top, w, h)
    b = V.ax(fig, xb, y_top, wb, h)
    for ax_ in (a, b):
        ax_.set_ylim(h, 0)
        V.clean(ax_, left=False)
    a.axvspan(-0.3, 0.3, color="#ECE7DF", lw=0, zorder=0)
    V.zero(a, horizontal=True)
    a.set_xlim(-1.1, 3.2)
    a.set_xticks([-1, 0, 1, 2, 3], ["−1", "0", "1", "2", "3"])
    a.set_xlabel(f"effect on synchrony ({DZ})")
    b.set_xscale("log")
    b.set_xlim(0.9, 12)
    b.set_xticks([1, 3, 10], ["1", "3", "10"])
    b.xaxis.set_minor_locator(plt.NullLocator())
    b.axvspan(3, 12, color="#ECE7DF", lw=0, zorder=0)
    b.set_xlabel("BF$_{01}$")
    tr = a.get_yaxis_transform()
    lx = (x_lab - x_ax) / w
    for r, yy in zip(rows, ys):
        yc = yy + dy / 2
        if r[0] == "group":
            a.text(lx, yc + 0.02, r[1], transform=tr, fontweight="bold", color=INK, va="center")
        elif r[0] == "row":
            _, lab, unit, dz, ci, n, bf_ = r
            ci = ci if ci is not None else dz_ci_from(dz, n, 0.95)
            sig = ci[0] > 0 or ci[1] < 0
            col = INK if sig else SAND.dark
            V.estimate(a, yc, dz, ci, col, ms=5.4, lw=1.2, filled=sig, face=None)
            a.text(lx, yc, lab.strip(), transform=tr, color=INK, va="center")
            if lab.startswith("  "):
                a.text(lx - 0.06, yc, "", transform=tr)
            a.text(-0.03, yc, unit, transform=tr, color=INK3, va="center", ha="right")
            if bf_ is not None:
                b.plot([1, bf_], [yc, yc], color=SAND.light, lw=3.4, solid_capstyle="butt", zorder=1)
                b.plot(bf_, yc, "o", ms=4.6, color=SAND.dark, zorder=2)
                b.text(13, yc, f"{bf_:.1f}", color=INK2, va="center", ha="left")
        elif r[0] == "pooled":
            pl = r[2]
            V.estimate(a, yc, pl["dz"], pl["ci95"], SAND.dark, marker="D", ms=6.0, lw=1.3, filled=False)
            a.text(lx, yc, r[1], transform=tr, color=INK, va="center", fontweight="bold")
            a.text(pl["ci95"][1] + 0.14, yc, f"{V.num(pl['dz'])} {V.ci_txt(pl['ci95'])}", color=SAND.dark, va="center")
    a.text(0, -0.03, f"|{DZ}| < 0.3", color=INK3, ha="center", va="bottom")
    b.text(np.sqrt(3 * 12), -0.03, "moderate", color=INK3, ha="center", va="bottom")
    return a, b, h


def fig3_attention(fig, x, y_top, w=1.55, h=1.60):
    """(b) attention: the attended story decoded from each listener's EEG, but no difference in synchrony."""
    E = pd.read_csv(TI / "cocktail_party_isc.csv")
    diff = E.assign(d=E.same - E.other).groupby("listener").d.mean()
    dec = pd.read_csv(TX / "attention_decoding.csv").set_index("listener").accuracy.reindex(diff.index) * 100
    e1 = J(P5 / "evidence_absence_icc.json")["absence"]["E1"]
    mg = 0.22
    a = V.ax(fig, x, y_top + mg + 0.04, w, h)
    tx_ = V.ax(fig, x, y_top, w, mg)
    rx = V.ax(fig, x + w + 0.04, y_top + mg + 0.04, mg, h)
    lim = float(np.abs(diff).max()) * 1.2
    a.axvspan(35, 50, color="#F2F0EB", lw=0, zorder=0)
    V.zero(a)
    a.scatter(dec, diff, s=16, color=SAND.base, edgecolors=PAPER, linewidths=0.5, zorder=3)
    (mx, hx), (my, hy) = V.mean_ci(dec), V.mean_ci(diff)
    a.plot([mx - hx, mx + hx], [my, my], color=INK, lw=1.2, zorder=4)
    a.plot([mx, mx], [my - hy, my + hy], color=INK, lw=1.2, zorder=4)
    a.plot(mx, my, "o", ms=5.6, mfc=PAPER, mec=INK, mew=1.1, zorder=5)
    a.set_xlim(35, 103)
    a.set_ylim(-lim, lim)
    a.set_xticks([50, 75, 100])
    a.set_yticks([-0.05, 0, 0.05], ["−0.05", "0", "0.05"])
    a.set_xlabel("attended story decoded (% runs)")
    a.set_ylabel("ISC, same − other story")
    a.text(42.5, lim * 0.96, "chance", color=INK2, ha="center", va="top", rotation=90)
    a.set_yticks([-0.05, 0, 0.05], ["−0.05", "0", "0.05"])
    gx = np.linspace(35, 100, 200)
    tx_.fill_between(gx, 0, V.kde(dec, gx, 0.35), color=INK, alpha=0.85, lw=0)
    tx_.set_xlim(35, 103)
    tx_.set_ylim(0, None)
    tx_.axis("off")
    gy = np.linspace(-lim, lim, 200)
    rx.fill_betweenx(gy, 0, V.kde(diff, gy, 0.45), color=SAND.light, lw=0)
    rx.set_ylim(-lim, lim)
    rx.set_xlim(0, None)
    rx.axis("off")
    V.text(fig, x + w + 0.42, y_top + mg + 0.30, f"decoded in {dec.mean():.0f}%", color=INK)
    V.text(fig, x + w + 0.42, y_top + mg + 0.47, "of runs", color=INK)
    V.text(fig, x + w + 0.42, y_top + mg + 0.80, "synchrony:", color=SAND.dark)
    V.text(fig, x + w + 0.42, y_top + mg + 0.97, f"{DZ} = {V.num(e1['dz'])}", color=SAND.dark)
    V.text(fig, x + w + 0.42, y_top + mg + 1.14, V.bf(e1["bf01_two_sided"]), color=SAND.dark)
    return a


def fig3_paradigm(fig, x, y_top):
    """Cocktail party: two stories in one mixture, each listener attends to one (schematic)."""
    c = V.canvas(fig, x, y_top, 2.30, 1.90)
    for k, (lab, col, yy) in enumerate((("story A", INK, 0.32), ("story B", INK3, 0.72))):
        _wave(c, 0.10, yy, 0.62, col, rev=bool(k), amp=0.08)
        c.text(0.10, yy - 0.16, lab, color=col, va="baseline")
        V.arrow(c, (0.80, yy), (1.10, 0.52), color=INK3, lw=0.7, ms=5)
    _head(c, 1.30, 0.52, r=0.13, fc=INKR.light, ec=INK)
    c.text(1.30, 0.86, "one mixture,\none story attended", ha="center", va="top", color=INK2, linespacing=1.15)
    c.text(0.10, 1.38, "Decoder: which story?", color=INK, va="baseline")
    c.text(0.10, 1.56, "Synchrony: same vs other story?", color=SAND.dark, va="baseline")
    return c


def figure3():
    fig = V.figure(5.50)
    V.panel(fig, 0.30, 0.20, "a", "Every synchrony test on one scale")
    fig3_profile(fig, 0.30, 3.05, 0.42, w=2.55, xb=5.88, wb=0.72)
    V.panel(fig, 0.30, 3.34, "b", "Attention: decoded, but not synchronised")
    fig3_paradigm(fig, 0.30, 3.48)
    fig3_attention(fig, 3.45, 3.46, w=1.90, h=1.30)
    V.save(fig, OUT, "Figure3")


def _meg_layout():
    return _illustration("meg_layout")["xy"]


def fig4_scatter(fig, x, y_top, xv, yv, rho, ci, xlim, title, sig, w=1.75, h=1.62):
    a = V.ax(fig, x, y_top, w, h)
    col = INK if sig else SAND.dark
    b1, b0 = np.polyfit(xv, yv, 1)
    xx = np.linspace(xlim[0] + 1, xlim[1] - 1, 60)
    rng = np.random.default_rng(3)
    boot = np.array([np.polyval(np.polyfit(xv[i], yv[i], 1), xx) for i in (rng.integers(0, len(xv), len(xv)) for _ in range(2000))])
    a.fill_between(xx, *np.percentile(boot, [2.5, 97.5], axis=0), color=INKR.tint if sig else SAND.tint, lw=0, zorder=0)
    a.plot(xx, b0 + b1 * xx, color=col, lw=1.1, zorder=1)
    jit = np.random.default_rng(1).uniform(-1.2, 1.2, len(xv))
    a.scatter(xv + jit, yv, s=16, color=INK if sig else SAND.base, edgecolors=PAPER, linewidths=0.5, zorder=2)
    a.set_xlim(*xlim)
    lo, hi = yv.min(), yv.max()
    a.set_ylim(lo - 0.06 * (hi - lo), hi + 0.32 * (hi - lo))       # headroom: the statistic sits above every point
    a.set_xlabel("comprehension (% correct)")
    a.set_ylabel("synchrony (ISC)")
    a.text(0.03, 0.98, f"ρ = {V.num(rho)}", transform=a.transAxes, color=col, ha="left", va="top", fontweight="bold" if sig else "normal",
           path_effects=V.halo(2.4), zorder=6)
    a.text(0.03, 0.87, V.ci_txt(ci), transform=a.transAxes, color=INK2, ha="left", va="top", path_effects=V.halo(2.4), zorder=6)
    V.text(fig, x, y_top - 0.08, title, color=INK2)
    return a


def fig4_meg_maps(fig, x, y_top, s=0.78):
    A = pd.read_csv(T7 / "meg_attention_sync_maps.csv")
    comp = J(T7 / "meg_attention_sync_components.json")
    xy = _meg_layout()
    for k in range(3):
        v = A[f"A_c{k + 1}"].to_numpy()
        v = v / np.abs(v).max()
        a = V.ax(fig, x, y_top + k * (s + 0.12), s, s)
        V.topomap(a, xy, v, cmap=V.MAP_FIELD, norm=Normalize(-1, 1), contours=4, sensors=False)
        V.text(fig, x + s + 0.06, y_top + k * (s + 0.12) + s * 0.45, f"component {k + 1}", color=INK)
        V.text(fig, x + s + 0.06, y_top + k * (s + 0.12) + s * 0.45 + 0.17, f"ISC {comp['isc_all_chapters'][k]:.3f}", color=INK2)


def fig4_links(fig, x_lab, x, y_top, w=2.6):
    """(d) Spearman correlation with comprehension of synchrony and of the word-level reference, per dataset and pooled,
    with their difference."""
    st = J(T78 / "pooled_word_level_contrast.json")          # Alice and pooled delta rho with up to 500 permutations
    fa = J(T78 / "pooled_correlations.json")["alice_fisher_ci"]
    pw = J(T78 / "pooled_correlations.json")["pooled_rho_comprehension_word_level"]
    B = J(T7 / "meg_attention_results.json")["B"]
    rows = [("Alice (EEG)", (fa["rho_comp_sync"]["rho"], fa["rho_comp_sync"]["ci95"]), (fa["rho_comp_dr_500_permutations"]["rho"], fa["rho_comp_dr_500_permutations"]["ci95"]),
             (st["alice_delta_rho_500_permutations"]["delta_rho"], st["alice_delta_rho_500_permutations"]["ci95"])),
            ("MEG attention", (B["rho_comp_sync"]["rho"], B["rho_comp_sync"]["ci95_fisher"]), (B["rho_comp_dr"]["rho"], B["rho_comp_dr"]["ci95_fisher"]),
             (B["delta_rho"]["observed"], B["delta_rho"]["ci95"])),
            ("pooled", (B["pooled"]["rho_comp_sync"], B["pooled"]["rho_ci95"]), (pw["rho"], pw["ci95"]),
             (st["pooled_delta_rho_500_permutations"]["delta_rho"], st["pooled_delta_rho_500_permutations"]["ci95"]))]
    a = V.ax(fig, x, y_top, w, 0.86)
    V.zero(a, horizontal=True)
    for i, (lab, sy, wl, dr) in enumerate(rows):
        yc = i
        V.estimate(a, yc - 0.15, sy[0], sy[1], INK if sy[1][0] > 0 else SAND.dark, ms=5.2, filled=sy[1][0] > 0,
                   marker="D" if lab == "pooled" else "o")
        V.estimate(a, yc + 0.15, wl[0], wl[1], WORDS.base, ms=5.2, filled=wl[1][0] > 0, marker="D" if lab == "pooled" else "o")
        a.text(-0.03, yc, lab, transform=a.get_yaxis_transform(), color=INK, ha="right", va="center")
        a.text(1.03, yc, f"Δρ = {V.num(dr[0])} {V.ci_txt(dr[1])}", transform=a.get_yaxis_transform(), color=INK if dr[1][0] > 0 else INK2,
               ha="left", va="center", fontweight="bold" if dr[1][0] > 0 else "normal")
    a.set_ylim(2.5, -0.5)
    a.set_xlim(-0.6, 0.8)
    a.set_xticks([-0.5, 0, 0.5], ["−0.5", "0", "0.5"])
    V.clean(a, left=False)
    a.set_xlabel("ρ with comprehension (95% CI)")
    V.text(fig, x, y_top - 0.10, "synchrony", color=INK, fontweight="bold")
    V.text(fig, x + 0.82, y_top - 0.10, "word level", color=WORDS.dark, fontweight="bold")
    return a


def figure4():
    Kl = pd.read_csv(TI / "alice_acoustic_removal_listeners.csv").dropna(subset=["comprehension"])
    fa = J(T78 / "pooled_correlations.json")["alice_fisher_ci"]["rho_comp_sync"]
    B = J(T7 / "meg_attention_results.json")["B"]
    part = pd.read_csv(T7 / "meg_attention_participants.csv").set_index("participant")
    sync = pd.read_csv(T7 / "meg_attention_sync_listeners.csv").set_index("listener")
    m = pd.DataFrame({"c": part.comprehension, "s": sync.sync}).dropna()
    fig = V.figure(5.10)
    V.panel(fig, 0.30, 0.20, "a", "Alice")
    fig4_scatter(fig, 0.72, 0.62, Kl.comprehension.to_numpy() * 100, Kl.isc.to_numpy(), fa["rho"], fa["ci95"], (20, 104),
                 "EEG · 47 listeners · 8-item quiz", True)
    V.panel(fig, 2.86, 0.20, "b", "MEG attention dataset")
    fig4_scatter(fig, 3.28, 0.62, m.c.to_numpy() * 100, m.s.to_numpy(), B["rho_comp_sync"]["rho"], B["rho_comp_sync"]["ci95_fisher"],
                 (55, 101), f"MEG · {B['n']} listeners · 30 questions", False)
    V.panel(fig, 5.36, 0.20, "c", "MEG synchrony components")
    fig4_meg_maps(fig, 5.40, 0.36)
    V.text(fig, 5.40, 3.10, "field patterns, sign arbitrary", color=INK3)
    V.panel(fig, 0.30, 3.40, "d", "Synchrony beside the word-level reference")
    fig4_links(fig, 0.30, 1.62, 3.92, w=3.30)
    V.save(fig, OUT, "Figure4")


def fig5_days(fig, x, y_top, w=1.70, h=2.10):
    """(a) every listener's synchrony in session 1 and session 2 (MEG-MASC), ranked, with Spearman rho."""
    r = J(T9 / "meg_masc_results.json")["test1"]
    sy = pd.read_csv(T9 / "meg_masc_sync_listeners.csv", dtype={"listener": str}).pivot(index="listener", columns="session", values="sync").dropna()
    a = V.ax(fig, x, y_top, w, h)
    order = sy[0].rank().to_numpy()
    cmap = plt.get_cmap("Greys")
    for (lid, row), rk in zip(sy.iterrows(), order):
        c = cmap(0.30 + 0.62 * (rk - 1) / (len(sy) - 1))
        a.plot([0, 1], [row[0], row[1]], color=c, lw=1.0, zorder=2)
        a.plot([0, 1], [row[0], row[1]], "o", ms=4.2, color=c, mec=PAPER, mew=0.5, zorder=3)
    a.set_xlim(-0.25, 1.25)
    a.set_xticks([0, 1], ["day 1", "day 2"])
    a.tick_params(axis="x", length=0, pad=3)
    a.spines["bottom"].set_visible(False)
    a.set_ylabel("synchrony (ISC)")
    p1 = r["primary"]
    a.text(0.5, 1.07, f"ρ = {V.num(p1['rho'])} {V.ci_txt(p1['ci95_fisher'])}", transform=a.transAxes, ha="center", va="bottom", color=INK,
           fontweight="bold")
    a.text(0.5, 1.00, f"partial ρ = {V.num(r['partial']['partial_rho'])}, 7 covariates", transform=a.transAxes, ha="center", va="bottom",
           color=INK2)
    return a


def fig5_partition(fig, x_lab, x, y_top, w=2.6, h=1.05):
    """(b) the share of single-segment variance that belongs to the listener: within a session (Alice, Aalborg) and, in
    MEG-MASC, the share stable across days and the further share stable within a day."""
    icc = J(P5 / "evidence_absence_icc.json")["icc"]
    m9 = J(T9 / "meg_masc_results.json")["test1"]["icc"]
    rows = [("Alice, one session", [(icc["alice_isc"]["icc_listener"], icc["alice_isc"]["ci95"], INK)]),
            ("Aalborg, one session", [(icc["aalborg_isc"]["icc_listener"], icc["aalborg_isc"]["ci95"], INK)]),
            ("MEG-MASC, two days", [(m9["across_days"]["share"], m9["across_days"]["ci95"], INK),
                                    (m9["within_day_only"]["share"], m9["within_day_only"]["ci95"], INKR.mid)])]
    a = V.ax(fig, x, y_top, w, h)
    for i, (lab, parts) in enumerate(rows):
        left = 0.0
        a.barh(i, 1.0, left=0, height=0.52, color=INKR.tint, edgecolor="none", zorder=0)
        for k, (v, ci, col) in enumerate(parts):
            a.barh(i, v, left=left, height=0.52, color=col, edgecolor=PAPER, linewidth=0.8, zorder=1)
            lo, hi = ci if k == 0 else m9["within_day_total"]["ci95"]        # the second interval is that of the total
            a.plot([lo, hi], [i + 0.34 + 0.08 * k, i + 0.34 + 0.08 * k], color=col, lw=1.0, solid_capstyle="butt", zorder=2)
            a.text(left + v / 2, i, f"{v:.2f}", ha="center", va="center", color=V.ink_on(col), fontweight="bold", zorder=3)
            left += v
        a.text(-0.02, i, lab, transform=a.get_yaxis_transform(), ha="right", va="center", color=INK)
    a.set_ylim(2.55, -0.5)
    a.set_xlim(0, 1)
    a.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    V.clean(a, left=False)
    a.set_xlabel("share of single-segment variance")
    yk = y_top - 0.16
    for xx, col, lab in ((x, INK, "listener (stable across days)"), (x + 1.72, INKR.mid, "+ within a day")):
        c = V.canvas(fig, xx, yk - 0.07, 0.16, 0.12)
        c.add_patch(Rectangle((0, 0), 0.14, 0.12, color=col, lw=0))
        V.text(fig, xx + 0.20, yk + 0.03, lab, color=INK2)
    return a


def fig5_cross(fig, x, y_top, which, s=1.40):
    Jj = J(TI / "generality.json")
    if which == "aalborg":
        b = pd.read_csv(TI / "aalborg_isc_speech_space.csv").groupby(["subject", "condition"]).isc.mean().unstack()
        xv, yv, xl, yl, rho = b[["SD", "SF"]].mean(axis=1), b["MC"], "speech (ISC)", "solo cello (ISC)", Jj["aalborg"]["rho"]
    else:
        cj = pd.read_csv(TI / "generality_cantisong_listener_isc.csv")
        xv, yv, xl, yl, rho = cj.speech, cj.melody, "spoken lyrics (ISC)", "hummed melodies (ISC)", Jj["cantisong"]["rho"]
    sig = which == "aalborg"
    a = V.ax(fig, x, y_top, s, s)
    if sig:
        b1, b0 = np.polyfit(xv, yv, 1)
        xx = np.linspace(min(xv), max(xv), 20)
        a.plot(xx, b0 + b1 * xx, color=INK, lw=1.0, zorder=1)
    a.scatter(xv, yv, s=16, color=INK if sig else SAND.base, edgecolors=PAPER, linewidths=0.5, zorder=2)
    a.set_xlabel(xl)
    a.set_ylabel(yl)
    a.text(0.04, 0.97, f"ρ = {V.num(round(rho, 2) + 0.0)}", transform=a.transAxes, va="top", color=INK if sig else SAND.dark,
           fontweight="bold" if sig else "normal")
    a.locator_params(axis="both", nbins=4)
    return a


def fig5_lists(fig, x, y_top, w=1.30):
    t2 = J(T9 / "meg_masc_results.json")["test2"]
    rows = [("all activity", t2["sync_2b_all"]), ("beyond the\nacoustics", t2["sync_2b_acoustics_removed"])]
    a = V.ax(fig, x, y_top, w, 0.86)
    a.axvspan(-0.5, 0.5, color="#ECE7DF", lw=0, zorder=0)
    V.zero(a, horizontal=True)
    for i, (lab, d) in enumerate(rows):
        lo, hi = d["dz_ci95"]
        sig = lo > 0
        V.estimate(a, i, d["dz"], (lo, hi), INK if sig else SAND.dark, ms=5.4, filled=sig)
        a.text(-0.03, i, lab, transform=a.get_yaxis_transform(), ha="right", va="center", color=INK, linespacing=1.0)
    d = t2["sync_2b_acoustics_removed"]
    a.text(d["dz_ci95"][1] + 0.12, 1, V.bf(d["bf01_two_sided"]), color=SAND.dark, va="center")
    a.set_ylim(1.6, -0.6)
    a.set_xlim(-0.6, 2.5)
    a.set_xticks([0, 1, 2])
    V.clean(a, left=False)
    a.set_xlabel(f"sentences − lists ({DZ})")
    return a


def figure5():
    fig = V.figure(5.12)
    V.panel(fig, 0.30, 0.20, "a", "Stable across days")
    fig5_days(fig, 0.82, 0.78)
    V.text(fig, 0.82, 0.38, "MEG-MASC · 22 listeners", color=INK2)
    V.panel(fig, 3.05, 0.20, "b", "The listener's share of synchrony")
    fig5_partition(fig, 3.05, 4.40, 0.74, w=2.55, h=1.80)
    V.panel(fig, 0.30, 3.34, "c", "Speech and cello")
    fig5_cross(fig, 0.82, 3.56, "aalborg", s=1.24)
    V.panel(fig, 2.60, 3.34, "d", "Spoken and hummed")
    fig5_cross(fig, 3.12, 3.56, "cantisong", s=1.24)
    V.panel(fig, 4.88, 3.34, "e", "Sentences vs word lists")
    fig5_lists(fig, 5.72, 3.74, w=1.25)
    V.save(fig, OUT, "Figure5")


def _space_pattern(cond):
    """Display only: the first correlated component (delta band) fitted on all 18 CantiSong songs in one version (speech:
    the spoken lyrics; melody: the hummed melody), and its forward model, Cz positive (the sign is arbitrary). Computed by
    figures/illustration_data.py."""
    d = _illustration(f"space_{cond}")
    return d["chans"].tolist(), d["A"]


def _above(a, x, lines, colors, y0=1.03, dy=0.075, bold_first=True):
    tp = a.get_xaxis_transform()
    for k, (s, col) in enumerate(zip(lines, colors)):
        a.text(x, y0 + (len(lines) - 1 - k) * dy, s, transform=tp, ha="center", va="baseline", color=col,
               fontweight="bold" if (k == 0 and bold_first) else "normal")


def fig6_spaces(fig, x, y_top, s=0.74):
    """(a, left) the two kinds of space, by the scalp pattern of their first component (display)."""
    for k, (cond, ramp, name, sub) in enumerate((("speech", WORDS, "speech-defined", "fitted on spoken lyrics"),
                                                  ("melody", MELODY, "melody-defined", "fitted on hummed melodies"))):
        chans, A = _space_pattern(cond)
        y = y_top + k * 1.10
        a = V.ax(fig, x + 0.18, y, s, s)
        V.topomap(a, V.eeg_xy(chans), A / np.abs(A).max(), cmap=V.MAP_FIELD, norm=Normalize(-1, 1), contours=5, sensor_size=0.9)
        V.text(fig, x + 0.18 + s / 2, y + s + 0.15, name, ha="center", color=ramp.dark, fontweight="bold")
        V.text(fig, x + 0.18 + s / 2, y + s + 0.29, sub, ha="center", color=INK2)


def fig6_songs(fig, x, y_top, w=2.55, h=1.95, xl=4.78, wl=1.55):
    """(a) per song, the sung − hummed difference in speech- and melody-defined dimensions (dot pairs), ordered by L, and
    each song's L = speech − melody on the same rows."""
    c = pd.read_csv(TI / "song_excess_songs.csv", dtype={"song": str}).groupby("song")[["sp_sung", "sp_hum", "mel_sung", "mel_hum", "L"]].mean()
    c["dsp"], c["dmel"] = c.sp_sung - c.sp_hum, c.mel_sung - c.mel_hum
    c = c.sort_values("L", ascending=False)
    T = J(TI / "song_excess.json")["primary"]
    Lp, bf = T["primary_items_L"], J(TI / "song_excess_bounds.json")["primary_items_L"]["bf_plus0"]
    n = len(c)
    a = V.ax(fig, x, y_top, w, h)
    for i, (_, r) in enumerate(c.iterrows()):
        a.plot([r.dmel, r.dsp], [i, i], color="#C9CDD2", lw=1.4, solid_capstyle="round", zorder=1)
        a.plot(r.dmel, i, "o", ms=4.8, mfc=MELODY.base, mec=PAPER, mew=0.5, zorder=3)
        a.plot(r.dsp, i, "o", ms=4.8, mfc=WORDS.base, mec=PAPER, mew=0.5, zorder=3)
    V.zero(a, horizontal=True)
    a.set_ylim(n - 0.4, -0.6)
    a.set_xlim(-0.006, 0.15)
    a.set_xticks([0, 0.05, 0.1, 0.15], ["0", "0.05", "0.1", "0.15"])
    a.set_yticks([])
    V.clean(a, left=False)
    a.set_xlabel("sung − hummed ISC")
    a.set_ylabel("18 songs, ordered by L", labelpad=4)
    n_both = int(((c.dsp > 0) & (c.dmel > 0)).sum())
    a.text(0.5, 1.10, f"sung > hummed in {n_both} of {n} songs, in both", transform=a.transAxes, ha="center", va="baseline",
           color=SUNG.dark, fontweight="bold")
    tp = a.transAxes
    for k, (ramp, lab) in enumerate(((WORDS, "speech-defined"), (MELODY, "melody-defined"))):
        xx = 0.06 + 0.50 * k
        a.plot(xx, 1.035, "o", ms=4.8, mfc=ramp.base, mec=PAPER, mew=0.5, transform=tp, clip_on=False)
        a.text(xx + 0.04, 1.035, lab, transform=tp, ha="left", va="center", color=ramp.dark)
    # each song's L on the same rows
    b = V.ax(fig, xl, y_top, wl, h)
    b.barh(np.arange(n), c.L, height=0.62, color=SAND.mid, lw=0, zorder=2)
    V.zero(b, horizontal=True)
    b.set_ylim(n - 0.4, -0.6)
    b.set_xlim(-0.05, 0.05)
    b.set_xticks([-0.04, 0, 0.04], ["−0.04", "0", "0.04"])
    b.set_yticks([])
    V.clean(b, left=False)
    b.set_xlabel("L = speech − melody")
    _above(b, 0.0, [f"L: {DZ} = {V.num(Lp['dz'])} {V.ci_txt(Lp['dz_ci95'])}",
                    f"{Lp['n_positive']} of {n} > 0 · BF$_{{+0}}$ = {bf:.2f}"], [SAND.dark, INK2], y0=1.035, dy=0.075)
    return a, b


def fig6_understood(fig, x, y_top, w=2.35, h=1.75):
    """(b) understood (Danish) − foreign (Finnish) ISC in the speech-defined space: per listener for speech and for songs,
    and per pair of songs (mean over listeners)."""
    raw = pd.read_csv(TI / "aalborg_isc_speech_space.csv")
    per = raw.groupby(["subject", "condition"]).isc.mean().unstack()
    B = J(TI / "understood_foreign.json")["primary_broderick_space"]
    pr = raw.assign(k=raw.stimulus.str.split("_").str[0]).groupby(["k", "condition"]).isc.mean().unstack()
    dp = (pr.MD - pr.MF).to_numpy()
    assert np.allclose(np.round(dp, 4), B["item_paired_MD-MF"]["diffs"])        # the song-pair differences
    sp, so = (per.SD - per.SF).to_numpy(), (per.MD - per.MF).to_numpy()
    a = V.ax(fig, x, y_top, w, h)
    a.set_xlim(-0.55, 2.45)
    a.set_ylim(-0.17, 0.46)
    V.zero(a)
    for k, (v, ramp, st) in enumerate(((sp, WORDS, B["SD-SF"]), (so, SUNG, B["MD-MF"]))):
        V.swarm(a, k - 0.06, v, ramp.base, size_pt=4.2, max_pt=11)
        assert abs(np.mean(v) - st["mean"]) < 1e-9
        m, hh = V.mean_ci(v)
        V.estimate(a, k + 0.30, m, (m - hh, m + hh), ramp.dark, horizontal=False, ms=5.6, filled=False)
        _above(a, k, [f"{DZ} = {V.num(st['dz'])}", f"{st['n_positive']} of {st['n']} > 0"], [ramp.dark, INK2])
    from figures import main_style as PD
    off = PD.pts_to_data_x(a, PD.swarm_offsets(a, dp, 5.4, max_pt=14))
    a.plot(2.0 + off, dp, "D", ms=4.6, mfc=SUNG.light, mec=SUNG.dark, mew=0.8, ls="none", zorder=3)
    _above(a, 2, [rf"$\mathit{{p}}$ = {B['item_paired_MD-MF']['p_one_sided']:.2f}", f"{int((dp > 0).sum())} of {len(dp)} > 0"],
           [INK2, INK2], bold_first=False)
    a.set_xticks([0, 1, 2], ["speech", "songs", "song pairs"])
    for lab_, col in zip(a.get_xticklabels(), (WORDS.dark, SUNG.dark, SUNG.dark)):
        lab_.set_color(col)
    a.tick_params(axis="x", length=0, pad=4)
    a.spines["bottom"].set_visible(False)
    a.set_ylabel("understood − foreign ISC")
    return a


def fig6_q2(fig, x, y_top, w=2.35, h=1.75):
    """(c) each song's synchrony excess L and word-level sung − hummed contrast, divided by their SD over songs (so the
    mean is the item-level d_z), and their difference D with its paired bootstrap CI."""
    q2 = J(P6 / "song_contrast_checks.json")["primary"]["Q2"]
    Lp = J(TI / "song_excess.json")["primary"]["primary_items_L"]
    wl = J(P6 / "word_level.json")["by_songs"]["sung_minus_hummed"]
    Ls = pd.read_csv(TI / "song_excess_songs.csv", dtype={"song": str}).groupby("song").L.mean()
    it = pd.read_csv(P6 / "word_level_song_means.csv", dtype={"song_id": str}).set_index("song_id")
    ws = (it.song - it.melody).reindex(Ls.index)
    zL, zW = (Ls / Ls.std(ddof=1)).to_numpy(), (ws / ws.std(ddof=1)).to_numpy()
    assert abs(zL.mean() - Lp["dz"]) < 1e-6 and abs(zW.mean() - wl["dz"]) < 1e-6   # the means are the d_z of the result tables
    a = V.ax(fig, x, y_top, w, h)
    a.set_xlim(-0.55, 2.45)
    a.set_ylim(-2.4, 3.0)
    V.zero(a)
    for k, (v, st, col, face) in enumerate(((zL, Lp, SAND.dark, SAND.mid), (zW, wl, WORDS.dark, WORDS.base))):
        V.swarm(a, k - 0.06, v, face, size_pt=4.2, max_pt=11)
        lo, hi = st["dz_ci95"]
        V.estimate(a, k + 0.30, st["dz"], (lo, hi), col, horizontal=False, ms=5.6, filled=not (lo <= 0 <= hi))
        _above(a, k, [f"{DZ} = {V.num(st['dz'])}", f"{st['n_positive']} of {st['n']} > 0"], [col, INK2])
    lo, hi = q2["ci95"]
    V.estimate(a, 2.0, q2["D"], (lo, hi), INK, horizontal=False, ms=5.8, marker="D", filled=False)
    _above(a, 2, [f"$D$ = {V.num(q2['D'])}", V.p_txt(q2["p_one_sided"])], [INK, INK2])
    a.set_xticks([0, 1, 2], ["synchrony\nexcess L", "word level,\nsung − hummed", "difference"])
    for lab_, col in zip(a.get_xticklabels(), (SAND.dark, WORDS.dark, INK)):
        lab_.set_color(col)
    a.tick_params(axis="x", length=0, pad=4)
    a.spines["bottom"].set_visible(False)
    a.set_ylabel("per song, in SDs over songs")
    return a


def figure6():
    fig = V.figure(6.50)
    V.panel(fig, 0.30, 0.20, "a", "Sung vs hummed songs, in dimensions defined by speech and by melody")
    V.text(fig, 0.30, 0.40, "CantiSong · 18 songs; scalp patterns of the first component of each space (display)",
           color=INK2)
    fig6_spaces(fig, 0.30, 0.86)
    fig6_songs(fig, 1.92, 0.86)
    V.panel(fig, 0.30, 3.58, "b", "Understood vs foreign, in speech and in song")
    V.text(fig, 0.30, 3.78, "Aalborg · 18 listeners · Danish − Finnish", color=INK2)
    fig6_understood(fig, 0.86, 4.20)
    V.panel(fig, 3.90, 3.58, "c", "Synchrony beside the word level")
    V.text(fig, 3.90, 3.78, "18 songs · each song's value over the SD across songs", color=INK2)
    fig6_q2(fig, 4.46, 4.20)
    V.save(fig, OUT, "Figure6")


MAIN = {"1": figure1, "2": figure2, "3": figure3, "4": figure4, "5": figure5, "6": figure6}
