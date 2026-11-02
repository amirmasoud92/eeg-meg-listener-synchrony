"""Panels of the main and supplementary figures that combine several analyses."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as ss
from matplotlib.patches import Rectangle

from figures import main_style as D, shared_panels as PM, supplementary_panels as SF, supplementary_style as S
from figures.main_style import BAND, FS, GLOBAL, INK, INK2, INK3, MELODY, PAPER, RULE, SUNG, WORDS

J = PM.J
P = PM.P
TI, P5, P6, T7, T78, T9 = PM.TI, PM.P5, PM.P6, PM.T7, PM.T78, PM.T9
SYNC_GRID = (
    [("Signal", "Figs. 2, 5e", GLOBAL, ["intact vs reversed", "sentences vs lists"]),
     ("State, signal fixed", "Fig. 3", GLOBAL, ["seeing the talker", "attention", "run comprehension", "engagement, α"]),
     ("Listener", "Figs. 4, 5", GLOBAL, ["comprehension", "stability", "across domains"]),
     ("Song", "Fig. 6", GLOBAL, ["understood vs foreign", "sung vs hummed"]),
     ("Suppl.", "Figs. S2, S3", GLOBAL, ["network states"]),
     ("Reference", "Figs. 4b, 6d", WORDS, ["word level"], 0.30)],
    [("Aalborg songs and speech", "EEG", 18, {5: "f", 7: "f", 8: "f", 9: "f", 11: "o"}),
     ("Etard and Reichenbach", "EEG", 11, {11: "o"}),
     ("Broderick: natural speech", "EEG", 19, {0: "f"}),
     ("Broderick: time-reversed", "EEG", 10, {0: "f"}),
     ("Broderick: speech in noise", "EEG", 21, {2: "f"}),
     ("Broderick: cocktail party", "EEG", 33, {3: "f", 4: "f"}),
     ("Alice", "EEG", 48, {5: "f", 6: "f", 7: "f", 12: "s"}),
     ("CantiSong", "EEG", 20, {8: "f", 10: "f", 12: "s"}),
     ("Two audiobooks, one attended", "MEG", 58, {6: "f", 12: "s"}),
     ("MEG-MASC, two sessions", "MEG", 27, {1: "f", 7: "f"})])
SONG_GRID = (
    [("Word level", "Figs. 2–4", WORDS, ["speech", "song", "seeing the talker", "attention", "comprehension", "sentence context",
                                         "across days"]),
     ("Melody", "Fig. 5", MELODY, ["notes"])],
    [("CantiSong", "EEG", 20, {0: "f", 1: "f", 7: "f"}),
     ("Alice", "EEG", 48, {0: "f", 4: "f"}),
     ("Speech in noise", "EEG", 21, {0: "f", 2: "f"}),
     ("Two audiobooks, one attended", "MEG", 58, {3: "f", 4: "f"}),
     ("MEG-MASC, two sessions", "MEG", 27, {0: "f", 5: "f", 6: "f"})])
SYNC_ROWS = {"H1", "H1b", "H2", "H2b", "HR1", "A", "B", "D", "pooled", "C", "E1", "E2", "F1", "I1", "I3", "TH3a", "K1", "J1",
             "B (MEG attention)", "1 (MEG-MASC)"}


def figS1_marked(paper: str, save):
    """Every primary test of the project, with
    the tests reported in `paper` ("synchrony" or "song") in full colour and the others faded."""
    from figures.supplementary_style import LANG, MUSIC, SONG, SYNC, W, axes_in
    L = lambda f: J(SF.T / f)
    inf = L("aalborg_network_tests.json")["confirmatory"]
    hr1 = L("etard_contrast.json")["primary"]
    A = L("intact_reversed.json")["primary"]["pooled"]["runs_forward_minus_reversed"]
    B = L("understood_foreign.json")["primary_broderick_space"]["MD-MF"]
    C = L("song_excess.json")["primary"]["primary_items_L"]
    Dd = L("speech_in_noise.json")["primary_items_D"]
    syn = L("evidence_synthesis.json")
    E, F, G, H, I = (L(n) for n in ("cocktail_party.json", "acoustic_removal.json", "song_word_level.json", "alice.json", "engagement_alpha.json"))
    Jj, K, Ll, M = (L(n) for n in ("generality.json", "alice_acoustic_removal.json", "song_melody_models.json", "song_language_models.json"))
    Pp = L("song_word_level_checks.json")
    r7, r9 = L("meg_attention_results.json"), L("meg_masc_results.json")
    rj, rk = Jj["pooled"]["rho"], K["rho_comprehension_isc_res"]
    rho = lambda r, n: {"dz": r, "dz_ci95": SF.rho_ci(r, n)}
    p1 = r9["test1"]["primary"]
    rows = [  # (key, label, estimate, units, colour); key None for a group header
        (None, "Network states (Aalborg)", None, None, SYNC),
        ("H1", "H1  understood lyrics → speech-like states", inf["H1"], "18 listeners", SYNC),
        ("H1b", "H1b  comprehension-axis alignment", inf["H1b"], "18 listeners", SYNC),
        ("H2", "H2  capture index, intrinsic dynamics", inf["H2"], "18 listeners", SYNC),
        ("H2b", "H2b  fronto-temporal Granger influence", inf["H2b"], "18 listeners", SYNC),
        (None, "Network states (Etard replication)", None, None, SYNC),
        ("HR1", "HR1  within-session Δ", hr1, "11 listeners", SYNC),
        (None, "δ synchrony and understanding", None, None, SYNC),
        ("A", "A  forward > reversed speech", A, "20 runs", SYNC),
        ("B", "B  understood > foreign song", B, "18 listeners", SYNC),
        ("D", "D  audiovisual superadditivity", Dd, f"{Dd['n']} passages", SYNC),
        ("pooled", "pooled A, B, D (random effects)", {"dz": syn["random_effects"]["dz"], "dz_ci95": syn["random_effects"]["ci95_hksj"]}, "k = 3", SYNC),
        ("C", "C  sung > hummed, speech- vs melody-space", C, "18 songs", SYNC),
        (None, "δ synchrony with the signal fixed, beyond acoustics", None, None, SYNC),
        ("E1", "E1  same- > other-story synchrony", E["runs_same_minus_other"], "30 runs", SYNC),
        ("E2", "E2  comprehension → synchrony (within listener)", E["graded_attended_score_vs_isc_same"], "33 listeners", SYNC),
        ("F1", "F1  forward > reversed after acoustic removal", F["primary"], "20 runs", SYNC),
        ("I1", "I1  rated engagement → synchrony", I["aalborg"]["all"], "18 listeners", SYNC),
        ("I3", "I3  α state → lower synchrony", I["alice"]["within_alpha_isc_negative"], "48 listeners", SYNC),
        (None, "Synchrony across listeners (ρ)", None, None, SYNC),
        ("TH3a", "TH3a  ρ(comprehension, synchrony)", rho(H["rho_comprehension_isc"], 47), "47 listeners, ρ", SYNC),
        ("K1", "K1  ρ(comprehension, synchrony beyond acoustics)", rho(rk, K["n"]), f"{K['n']} listeners, ρ", SYNC),
        ("J1", "J1  ρ(speech synchrony, non-linguistic synchrony)", rho(rj, 35), "2 datasets, ρ", SYNC),
        (None, "Word predictability", None, None, LANG),
        ("G1", "G1  spoken lyrics", G["spoken_all"], "20 listeners", LANG),
        ("G2", "G2  sung lyrics", G["sung_all"], "20 listeners", SONG),
        ("G2h", "hummed control (sung timing)", G["hummed_control"], "20 listeners", MUSIC),
        ("TH1", "TH1  speech, Alice (replication)", H["delta_r_gpt2"], f"{H['delta_r_gpt2']['n']} listeners", LANG),
        ("TH2", "TH2  ρ(comprehension, word-level Δr)", rho(H["rho_comprehension_delta_r"], 47), "47 listeners, ρ", LANG),
        ("S1s", "S1  sung lyrics, by songs", Pp["items_sung"], "18 songs", SONG),
        ("S1p", "S1  spoken lyrics, by songs", Pp["items_spoken"], "18 songs", LANG),
        ("S2", "S2  sung > hummed, by songs", Pp["items_sung_minus_hummed"], "18 songs", SONG),
        ("A1", "A1  lexical entropy, speech (Alice)", Pp["alice_entropy"], "48 listeners", LANG),
        ("A3", "A3  lexical entropy, sung lyrics", Pp["sung_entropy"], "20 listeners", SONG),
        ("M1p", "M1  GPT-2 representations, spoken", M["gpt_mean_speech"], "20 listeners", LANG),
        ("M1s", "M1  GPT-2 representations, sung", M["gpt_mean_song"], "20 listeners", SONG),
        ("2b", "2b  speech in noise, audiovisual − audio-only", L("speech_in_noise_word_level.json")["primary_av_minus_a"], "21 listeners", LANG),
        (None, "Melody", None, None, MUSIC),
        ("L1s", "L1  melodic surprisal, sung", Ll["L1_melodic_sung"], "20 listeners", SONG),
        ("L1h", "L1  melodic surprisal, hummed", Ll["L1_melodic_hummed"], "20 listeners", MUSIC),
        ("L2", "L2  melodic surprise reduces lexical prediction", Ll["L2_low_minus_high"], "20 listeners", INK2),
        (None, "MEG attention", None, None, LANG),
        ("A (MEG attention)", "A  word predictability, attended − ignored", r7["A"]["primary"], f"{r7['A']['primary']['n']} listeners", LANG),
        ("B (MEG attention)", "B  Δρ, synchrony − word level (comprehension)", {"dz": r7["B"]["delta_rho"]["observed"], "dz_ci95": r7["B"]["delta_rho"]["ci95"]},
         f"{r7['B']['n']} listeners", SYNC),
        (None, "MEG-MASC", None, None, LANG),
        ("1 (MEG-MASC)", "1  synchrony stable across days, ρ(session 1, session 2)", {"dz": p1["rho"], "dz_ci95": p1["ci95_fisher"]},
         f"{p1['n']} listeners, ρ", SYNC),
        ("2 (MEG-MASC)", "2  word predictability, sentences − word lists (matched)", r9["test2"]["primary"], f"{r9['test2']['primary']['n']} listeners", LANG)]
    mine = (lambda key: key in SYNC_ROWS) if paper == "synchrony" else (lambda key: key not in SYNC_ROWS)
    XMIN, XMAX = -1.5, 3.5
    H_ = 0.235 * len(rows) + 1.3
    fig = plt.figure(figsize=(W, H_))
    ax = axes_in(fig, 2.75, 0.45, 2.55, H_ - 1.25)
    ax.axvspan(-0.5, 0.5, color=S.BAND, lw=0, zorder=0)
    S.zero(ax, horizontal=True)
    n = len(rows)
    for i, (key, name, s_, unit, c) in enumerate(rows):
        y = n - 1 - i
        if key is None:
            ax.text(-1.06, y, name, fontsize=6.2, fontweight="bold", color=c, va="center", ha="left", transform=ax.get_yaxis_transform())
            continue
        al, ink = (1.0, S.INK) if mine(key) else (0.28, S.INK3)
        lo, hi = s_["dz_ci95"]
        pooled = key == "pooled"
        clo, chi = max(lo, XMIN), min(hi, XMAX)
        ax.plot([clo, chi], [y, y], color=c, lw=1.0, solid_capstyle="butt", zorder=2, alpha=al)
        for edge, clipped, mk in ((clo, lo < XMIN, "<"), (chi, hi > XMAX, ">")):
            if clipped:
                ax.scatter([edge], [y], marker=mk, s=16, color=c, zorder=3, linewidths=0, alpha=al)
        ax.scatter([s_["dz"]], [y], s=26 if pooled else 15, marker="D" if pooled else "o", color=c, zorder=4, linewidths=0, alpha=al)
        ax.text(-1.0, y, name, fontsize=5.8, color=ink, va="center", ha="left", transform=ax.get_yaxis_transform())
        ax.text(1.03, y, f"{S.num(s_['dz'])} [{S.num(lo)}, {S.num(hi)}]", fontsize=5.4, color=S.INK2 if mine(key) else S.INK3, va="center",
                ha="left", transform=ax.get_yaxis_transform())
        ax.text(1.52, y, unit, fontsize=5.4, color=S.INK3, va="center", ha="left", transform=ax.get_yaxis_transform())
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([-1, 0, 1, 2, 3])
    ax.set_xlabel("effect size (dz, or ρ where marked) with 95 % CI; positive = predicted direction; band: |dz| < 0.5")
    ax.text(1.03, n - 0.1, "estimate [95 % CI]", fontsize=5.6, color=S.INK2, va="bottom", ha="left", transform=ax.get_yaxis_transform())
    ax.text(1.52, n - 0.1, "units", fontsize=5.6, color=S.INK2, va="bottom", ha="left", transform=ax.get_yaxis_transform())
    S.fig_heading(fig, 0.3, H_ - 0.25, "", "Every primary test in the project",
                  "primary tests of the 32 analyses and their secondary estimates; this paper's tests in full colour, the project's other tests faded")
    fig.text(0.3 / W, (H_ - 0.52) / H_, "colours: graphite synchrony and network states; blue words; red sung; ochre melody and hummed",
             fontsize=S.FS - 0.3, color=S.INK2, va="baseline")
    save(fig, "FigureS1")


def figS_further(save):
    """Synchrony Supplementary Fig. 5, one row: (a) the comprehension-synchrony correlation in Alice over bands and numbers
    of components; (b) the capture ratio and (c) L for each correlated component.
    Understood vs foreign speech is main Fig. 6b."""
    from figures.supplementary_style import INK3 as I3, MUSIC, SONG, SYNC, W, axes_in
    grid = J(TI / "alice_acoustic_removal.json")["R_TH3a_grid"]
    keys = list(grid)
    df = pd.read_csv(TI / "song_excess_songs.csv")
    item = df.groupby("song")[["cap_sung", "cap_hum"]].mean()
    C = J(TI / "song_excess.json")
    comp = pd.read_csv(TI / "song_excess_components.csv").pivot(index="song", columns="component", values="L")
    fig = plt.figure(figsize=(W, 2.7))
    R, B0, H = 2.48, 0.62, 1.38
    # (a) Alice: the correlation over bands and numbers of components
    ax = axes_in(fig, 0.75, B0, 1.80, H)
    S.zero(ax)
    for i, k_ in enumerate(keys):
        S.multi_interval(ax, i, grid[k_]["rho"], S.rho_levels(grid[k_]["rho"], 47), SYNC, horizontal=False, s=14, lw=(3.2, 1.7, 0.8))
    ax.set_xticks(range(len(keys)), [k_.split("|")[1] for k_ in keys], fontsize=5.6)
    for j, band in enumerate(dict.fromkeys(k_.split("|")[0] for k_ in keys)):
        idx = [i for i, k_ in enumerate(keys) if k_.split("|")[0] == band]
        ax.text(np.mean(idx), -0.20, band.replace("-", "–") + " Hz", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=5.6, color=S.INK2)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylim(-0.2, 0.9)
    ax.set_ylabel("ρ(comprehension, synchrony)")
    S.fig_heading(fig, 0.3, R, "a", "Comprehension in Alice",
                  f"47 listeners; correlated components per band; all p$_{{FWE}}$ ≤ {max(v['p_fwe'] for v in grid.values()):.3f}")
    # (b) song: synchrony of the sung and hummed versions relative to the spoken version
    cx = axes_in(fig, 3.40, B0, 0.85, H)
    SF.paired(cx, item.cap_sung, item.cap_hum, SONG, MUSIC, ["sung", "hummed"])
    cx.axhline(1, color=I3, lw=0.6, ls=(0, (2, 2)))
    cx.set_ylabel("synchrony relative to\nthe spoken version")
    S.fig_heading(fig, 2.95, R, "b", "Capture ratio", "not acoustically controlled")
    # (c) song: L for each correlated component alone, with the cumulative estimates of the component grid
    bx = axes_in(fig, 5.00, B0, 2.00, H)
    S.zero(bx)
    cg = {g_["ncomp"]: g_ for g_ in C["grid"] if g_["band"] == "1-4"}
    for k in comp.columns:
        v = comp[k].dropna()
        m, h = S.mean_ci(v)
        S.multi_interval(bx, k, m, {0.5: (m - h * 0.674 / 1.96, m + h * 0.674 / 1.96), 0.8: (m - h * 1.282 / 1.96, m + h * 1.282 / 1.96),
                                    0.95: (m - h, m + h)}, SYNC if k == 1 else I3, horizontal=False, s=16)
        if k in cg:
            bx.text(k, 1.02, f"cumulative dz {cg[k]['dz']:.2f}\np$_{{FWE}}$ {cg[k]['p_fwe']:.3f}", transform=bx.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=5.2, color=S.INK2)
    bx.set_xticks(list(comp.columns))
    bx.tick_params(axis="x", length=0)
    bx.spines["bottom"].set_visible(False)
    bx.set_xlim(0.4, len(comp.columns) + 0.6)
    bx.set_xlabel("component (same rank in both spaces)")
    bx.set_ylabel("L of that component alone")
    S.fig_heading(fig, 4.55, R, "c", "L for each component", "song means, 50/80/95 % intervals")
    save(fig, "FigureS5")
