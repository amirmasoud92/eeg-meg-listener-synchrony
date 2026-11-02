"""Style of the supplementary figures and the helpers they share."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from figures.main_style import BAND, GLOBAL, INK, INK2, INK3, MELODY, RULE, SUNG, WORDS
from statistical_tests.inference import dz_ci_from

LANG, SONG, MUSIC = WORDS.base, SUNG.base, MELODY.base
SYNC, SYNC_L = GLOBAL.base, GLOBAL.light
LANG_L, SONG_L, MUSIC_L = WORDS.light, SUNG.light, MELODY.light
CORTEX = mpl.colors.LinearSegmentedColormap.from_list("cortex", ["#E2EFEC", "#86C0B7", "#2C7C73", "#0C3B37"])
VERSION = {"speech": (LANG, LANG_L, "spoken"), "song": (SONG, SONG_L, "sung"), "melody": (MUSIC, MUSIC_L, "hummed")}
W = 7.2    # figure width in inches (183 mm)
FS = 6.5   # base font size


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Source Sans Pro", "DejaVu Sans"],
        "font.size": FS, "axes.titlesize": FS + 0.5, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5,
        "legend.fontsize": FS - 0.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.labelpad": 2.5, "axes.titlepad": 4, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False,
        "lines.solid_capstyle": "round", "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "mathtext.default": "regular",
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",                                   # math (d_z, p, BF_01) in Arial, as in the main figures
    })


def axes_in(fig, x, y, w, h, **kw):
    """Axes placed in inches from the figure's bottom-left corner."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh], **kw)


def fig_heading(fig, x, y, letter, title, sub=None, letter_dx=0.16):
    """Panel letter, bold title and optional grey subtitle at fixed figure positions (inches), so that the
    headings of one row align exactly regardless of each panel's axes."""
    fw, fh = fig.get_size_inches()
    fig.text((x - letter_dx) / fw, y / fh, letter, fontsize=9, fontweight="bold", ha="left", va="baseline", color=INK)
    fig.text(x / fw, y / fh, title, fontsize=FS + 0.5, fontweight="bold", ha="left", va="baseline", color=INK)
    if sub:
        fig.text(x / fw, (y - 0.13) / fh, sub, fontsize=FS - 0.3, ha="left", va="baseline", color=INK2)


def mean_ci(x, level=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x.mean(), stats.t.ppf(0.5 + level / 2, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)


def rho_levels(r, n, levels=(0.5, 0.8, 0.95)):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return {lv: (np.tanh(z - stats.norm.ppf(0.5 + lv / 2) * se), np.tanh(z + stats.norm.ppf(0.5 + lv / 2) * se)) for lv in levels}


def multi_interval(ax, pos, est, lv, color, horizontal=True, s=18, lw=(4.2, 2.2, 0.9), hollow=False):
    """Nested 50/80/95 % intervals (thick to thin) with the point estimate."""
    for (level, (lo, hi)), w in zip(sorted(lv.items()), lw):
        if horizontal:
            ax.plot([lo, hi], [pos, pos], color=color, lw=w, solid_capstyle="butt", zorder=2)
        else:
            ax.plot([pos, pos], [lo, hi], color=color, lw=w, solid_capstyle="butt", zorder=2)
    xy = (est, pos) if horizontal else (pos, est)
    ax.scatter(*xy, s=s, facecolor="white" if hollow else color, edgecolor=color, linewidths=1.0, zorder=3)


def num(x, fmt=".2f"):
    """Number with a typographic minus sign."""
    return format(x, fmt).replace("-", "−")


def zero(ax, horizontal=False, **kw):
    kw = {"color": INK3, "lw": 0.6, "zorder": 0, **kw}
    (ax.axvline if horizontal else ax.axhline)(0, **kw)
