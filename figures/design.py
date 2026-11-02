"""Layout grid, scalp and cortex maps and the panel conventions of the main figures."""
from __future__ import annotations

import sys
from collections import namedtuple
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from scipy import stats

from figures import main_style as PD

Ramp = namedtuple("Ramp", "tint light mid base dark")
WORDS, SUNG, MELODY = PD.WORDS, PD.SUNG, PD.MELODY
INKR = Ramp("#F1F2F4", "#D2D6DB", "#8B939D", "#3B4450", "#1C232C")      # synchrony and other global measures
SAND = Ramp("#F7F4EF", "#E6DED2", "#BBAE9A", "#8E8170", "#5F5546")      # not significant / no change
INK, INK2, INK3 = "#1B2128", "#4E5660", "#6A7179"          # text contrast on white: 16, 7.4 and 4.9 (>= 4.5)
RULE, BAND, PAPER = "#D5D8DC", "#F2F0EB", "#FFFFFF"
VERSION = {"speech": (WORDS, "spoken"), "song": (SUNG, "sung"), "melody": (MELODY, "hummed")}
W = 7.2
FS = {"letter": 10.0, "head": 8.5, "text": 8.0}
DZ = r"$\mathit{d}_\mathrm{z}$"
MAP_SYNC = LinearSegmentedColormap.from_list("sync", ["#F2F7F6", "#C5E0DC", "#82B9B2", "#3E8D86", "#145F5B", "#073B3A"])
MAP_FIELD = LinearSegmentedColormap.from_list("field", ["#6E4214", "#B88A55", "#E9DDCB", "#F3F3F1", "#C9E1DD", "#5FA39B", "#0F5552"])
MAP_WORDS = LinearSegmentedColormap.from_list("words", [WORDS.tint, WORDS.light, WORDS.mid, WORDS.base, WORDS.dark])
p_txt = PD.p_txt
mean_ci = PD.mean_ci


def setup():
    t = FS["text"]
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial", "mathtext.default": "regular",
        "font.size": t, "axes.labelsize": t, "axes.titlesize": t, "xtick.labelsize": t, "ytick.labelsize": t, "legend.fontsize": t,
        "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "axes.labelpad": 3.0, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "figure.facecolor": PAPER, "axes.facecolor": "none", "savefig.facecolor": PAPER, "legend.frameon": False,
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "butt", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 600, "path.simplify": False,
    })


def figure(h: float):
    return plt.figure(figsize=(W, h))


def ax(fig, x, y_top, w, h, **kw):
    """Axes placed in inches from the figure's top-left corner."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, (fh - y_top - h) / fh, w / fw, h / fh], **kw)


def canvas(fig, x, y_top, w, h):
    """Axes whose data units are inches, y increasing downwards (for schematics)."""
    a = ax(fig, x, y_top, w, h)
    a.set_xlim(0, w)
    a.set_ylim(h, 0)
    a.axis("off")
    return a


def text(fig, x, y_top, s, **kw):
    fw, fh = fig.get_size_inches()
    kw = {"fontsize": FS["text"], "va": "baseline", **kw}
    return fig.text(x / fw, (fh - y_top) / fh, s, **kw)


def panel(fig, x, y_top, letter, head=None, color=INK, dx=0.17):
    """Panel letter and a short heading on one baseline (y_top is the baseline)."""
    if letter:
        text(fig, x - dx, y_top, letter, fontsize=FS["letter"], fontweight="bold", color=INK)
    if head:
        text(fig, x, y_top, head, fontsize=FS["head"], fontweight="bold", color=color)


def clean(a, left=True, bottom=True):
    a.spines["left"].set_visible(left)
    a.spines["bottom"].set_visible(bottom)
    if not left:
        a.tick_params(left=False, labelleft=False)
    if not bottom:
        a.tick_params(bottom=False, labelbottom=False)


def zero(a, horizontal=False, **kw):
    kw = {"color": INK3, "lw": 0.6, "zorder": 0, **kw}
    (a.axvline if horizontal else a.axhline)(0, **kw)


def halo(lw=2.2):
    return [pe.withStroke(linewidth=lw, foreground=PAPER)]


def num(x, fmt=".2f"):
    return format(x, fmt).replace("-", "−")


def ci_txt(ci, fmt=".2f"):
    return f"[{num(ci[0], fmt)}, {num(ci[1], fmt)}]"


def bf(v, kind="01"):
    return f"BF$_{{{kind}}}$ = {v:.1f}"


def kde(v, grid, bw=None):
    return stats.gaussian_kde(np.asarray(v, float), bw_method=bw)(grid)


def estimate(a, pos, est, ci, color, horizontal=True, filled=True, ms=5.6, lw=1.3, marker="o", z=4, face=None, ci50=None):
    seg = (lambda u, v: ([u, v], [pos, pos])) if horizontal else (lambda u, v: ([pos, pos], [u, v]))
    a.plot(*seg(*ci), color=color, lw=lw, solid_capstyle="butt", zorder=z)
    if ci50 is not None:
        a.plot(*seg(*ci50), color=color, lw=lw * 2.6, solid_capstyle="butt", zorder=z)
    xy = (est, pos) if horizontal else (pos, est)
    a.plot(*xy, marker=marker, ms=ms, mfc=(face or color) if filled else PAPER, mec=color, mew=1.1, zorder=z + 1, ls="none")


def swarm(a, x, v, color, size_pt=3.6, edge=PAPER, max_pt=None, z=3, mfc=None, lw=0.4):
    off = PD.pts_to_data_x(a, PD.swarm_offsets(a, v, size_pt, max_pt=max_pt))
    a.scatter(x + off, v, s=size_pt ** 2, facecolor=mfc or color, edgecolors=edge, linewidths=lw, zorder=z)
    return x + off


def arrow(a, p0, p1, color=INK2, lw=0.8, style="-|>", ms=7, z=2, **kw):
    p = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0, **kw)
    a.add_patch(p)
    return p


def colorbar(fig, x, y_top, w, h, cmap, norm, ticks, ticklabels=None, label=None, horizontal=True, label_side="top"):
    c = ax(fig, x, y_top, w, h)
    lo, hi = norm.vmin, norm.vmax
    g = np.linspace(lo, hi, 256)
    img = cmap(norm(g))[None] if horizontal else cmap(norm(g))[::-1, None]
    c.imshow(img, aspect="auto", extent=(lo, hi, 0, 1) if horizontal else (0, 1, lo, hi))
    for s in c.spines.values():
        s.set_visible(False)
    if horizontal:
        c.set_yticks([])
        c.set_xticks(ticks, ticklabels or [num(t, "g") for t in ticks])
        c.tick_params(axis="x", length=2.0, pad=1.5)
        if label:
            c.text(0.5 if label_side == "top" else -0.03, 1.25 if label_side == "top" else 0.5, label, transform=c.transAxes,
                   ha="center" if label_side == "top" else "right", va="bottom" if label_side == "top" else "center", color=INK2)
    else:
        c.set_xticks([])
        c.yaxis.tick_right()
        c.set_yticks(ticks, ticklabels or [num(t, "g") for t in ticks])
        c.tick_params(axis="y", length=2.0, pad=1.5)
        if label:
            c.text(0.5, 1.06, label, transform=c.transAxes, ha="center", va="bottom", color=INK2)
    return c


def _interp_map(xy, v, res=220, radius=1.0):
    from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator
    # anchor the rim with nearest-neighbour values so the interpolation covers the head disc
    ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    rim = np.c_[np.cos(ang), np.sin(ang)] * radius * 1.04
    nn = NearestNDInterpolator(xy, v)
    pts = np.r_[xy, rim]
    vals = np.r_[v, nn(rim)]
    g = np.linspace(-radius, radius, res)
    gx, gy = np.meshgrid(g, g)
    z = CloughTocher2DInterpolator(pts, vals)(gx, gy)
    z[gx ** 2 + gy ** 2 > radius ** 2] = np.nan
    return gx, gy, z


def head_outline(a, radius=1.0, color=INK2, lw=0.8, z=5):
    a.add_patch(Circle((0, 0), radius, fill=False, ec=color, lw=lw, zorder=z))
    a.add_patch(Polygon([(-0.11 * radius, 0.993 * radius), (0, 1.12 * radius), (0.11 * radius, 0.993 * radius)], closed=False,
                        fill=False, ec=color, lw=lw, zorder=z, joinstyle="round"))
    for s in (-1, 1):
        a.add_patch(Polygon([(s * 0.995 * radius, 0.13 * radius), (s * 1.06 * radius, 0.11 * radius), (s * 1.075 * radius, -0.12 * radius),
                             (s * 0.995 * radius, -0.17 * radius)], closed=False, fill=False, ec=color, lw=lw, zorder=z,
                            joinstyle="round"))


def topomap(a, xy, v, cmap=MAP_FIELD, norm=None, contours=6, sensors=True, sensor_size=1.4, res=220, lw_contour=0.35):
    """A scalp or sensor map: values ``v`` at 2-D positions ``xy`` (head radius 1, nose up)."""
    v = np.asarray(v, float)
    lim = np.nanmax(np.abs(v))
    norm = norm or Normalize(-lim, lim)
    gx, gy, z = _interp_map(xy, v, res)
    a.imshow(np.ma.masked_invalid(z), origin="lower", extent=(-1, 1, -1, 1), cmap=cmap, norm=norm, interpolation="bilinear",
             zorder=1).set_rasterized(True)
    if contours:
        lv = np.linspace(norm.vmin, norm.vmax, contours + 2)[1:-1]
        cs = a.contour(gx, gy, z, levels=lv, colors=INK, linewidths=lw_contour, alpha=0.45, zorder=2)
        for c, l in zip(cs.collections if hasattr(cs, "collections") else [cs], lv):
            pass
    if sensors:
        a.scatter(xy[:, 0], xy[:, 1], s=sensor_size, color=INK, alpha=0.55, linewidths=0, zorder=3)
    head_outline(a)
    a.set_xlim(-1.18, 1.18)
    a.set_ylim(-1.12, 1.22)
    a.set_aspect("equal")
    a.axis("off")


def eeg_xy(ch_names):
    """Azimuthal-equidistant 2-D positions of standard 10-05 electrodes (head radius 1 at the 10-20 equator)."""
    import mne
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    lut = {k.lower(): v for k, v in pos.items()}

    def proj(names):
        xyz = np.array([lut[c.lower()] for c in names]) - np.array([0.0, -0.017, 0.0])
        th = np.arccos(np.clip(xyz[:, 2] / np.linalg.norm(xyz, axis=1), -1, 1))
        ph = np.arctan2(xyz[:, 1], xyz[:, 0])
        return th, ph
    th, ph = proj(ch_names)
    ref = proj(["Fpz", "T7", "T8", "Oz"])[0].mean()
    return np.c_[th * np.cos(ph), th * np.sin(ph)] / ref


def brain(a, values, view, cmap, norm, faded=(), labels="dk18", outline="#FFFFFF", outline_lw=0.25, fade=0.70):
    """Cortical map on inflated fsaverage: the 18 analysis regions ('dk18') or every Desikan-Killiany label ('dk68')."""
    from figures import brain as BR
    from common.settings import load_paths
    fs_dir = load_paths().mne_data / "fsaverage"
    if labels == "dk68":
        from figures import shared_panels as PM
        orig = BR.surface
        BR.surface = PM._dk_surface
        try:
            BR.draw(a, values, fs_dir, view, cmap=cmap, norm=norm, faded=set(faded), outline=outline, outline_lw=outline_lw, fade=fade)
        finally:
            BR.surface = orig
    else:
        BR.draw(a, values, fs_dir, view, cmap=cmap, norm=norm, faded=set(faded), outline=outline, outline_lw=outline_lw, fade=fade)


def region_xy(region, view, labels="dk18"):
    """Projected 2-D centroid of one region on ``view`` (for leader lines)."""
    from figures import brain as BR
    from common.settings import load_paths
    fs_dir = load_paths().mne_data / "fsaverage"
    if labels == "dk68":
        from figures import shared_panels as PM
        name, hemi = region.rsplit("_", 1)
        return PM._label_xy(name, hemi, view)
    x, y, _ = BR.centroids(fs_dir, "inflated", view)[region]
    return x, y


def check(fig, min_pt=8.0, ignore=()):
    return PD.check_layout(fig, min_pt=min_pt, ignore=ignore)


def save(fig, out: Path, name: str, min_pt=8.0, ignore=()):
    out.mkdir(parents=True, exist_ok=True)
    issues = check(fig, min_pt, ignore)
    for s in issues:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"            # a Windows console cannot print every glyph
    fig.savefig(out / f"{name}.png", dpi=600, metadata={"Software": None})
    fig.savefig(out / f"{name}.pdf", dpi=600, metadata={"CreationDate": None, "Creator": None, "Producer": None})
    plt.close(fig)
    return issues


def link_diagram(fig, y0, contrasts, datasets, ref_color=None, height=3.2, xr=4.86, xl=3.66, key=None):
    """Datasets (left: name, modality, listeners) linked to the contrasts each isolates (right), grouped under headings
    with the figures that test them. contrasts: [(group, figures, [contrast, ...], colour or None)]; datasets:
    [(name, modality, n, {contrast index: kind})] with kind "f" (main text), "o" (supplementary) or "s" (reference).
    Datasets are ordered by the mean position of their contrasts, which keeps crossings few."""
    from matplotlib.patches import Rectangle
    c = canvas(fig, 0, y0, W, height)
    yy, dy, gap = 0.12, 0.148, 0.06
    ypos, heads, cols_of = [], [], []
    for name, ref, cols, col in contrasts:
        yy += gap
        colr = col or INK
        if len(cols) > 1:
            heads.append((name, ref, yy, True, colr))
            yy += dy
            for cc in cols:
                ypos.append(yy)
                cols_of.append((cc, colr))
                yy += dy
        else:
            heads.append((name, ref, yy, False, colr))
            ypos.append(yy)
            cols_of.append((None, colr))
            yy += dy
    h = yy
    for name, ref, y, multi, colr in heads:
        c.text(xr - 0.06 if multi else xr + 0.06, y, name, fontweight="bold", color=colr, va="center")
        c.text(7.02, y, ref, color=INK2, va="center", ha="right")
    for (lab, colr), y in zip(cols_of, ypos):
        if lab:
            c.text(xr + 0.06, y, lab, color=INK, va="center")
        c.plot(xr - 0.02, y, "o", ms=2.8, color=colr, zorder=4)
    n_max = max(d[2] for d in datasets)
    order = sorted(datasets, key=lambda d: np.mean([ypos[j] for j in d[3]]))
    yl = np.linspace(0.30, h - 0.10, len(order)) if len(order) > 1 else [h / 2]
    styles = {"f": (INK, "-", 0.62), "o": (INK3, (0, (2.5, 1.6)), 0.9), "s": (ref_color or INK, "-", 0.85)}
    for (nm, mod, n, links), y in zip(order, yl):
        c.text(0.30, y, nm, color=INK, va="center")
        c.text(2.30, y, mod, color=INK if mod == "MEG" else INK2, va="center", fontweight="bold" if mod == "MEG" else "normal")
        c.add_patch(Rectangle((2.68, y - 0.035), 0.55 * n / n_max, 0.07, color=INKR.light, lw=0, zorder=1))
        c.text(2.68 + 0.55 * n / n_max + 0.05, y, str(n), color=INK2, va="center")
        c.plot(xl, y, "o", ms=2.8, color=INK, zorder=4)
        for j, kind in links.items():
            yt = ypos[j]
            xx = np.linspace(xl, xr - 0.02, 60)
            u = (xx - xl) / (xr - 0.02 - xl)
            col, ls, al = styles[kind]
            if kind == "f" and cols_of[j][1] != INK:
                col = cols_of[j][1]
            c.plot(xx, y + (yt - y) * (3 * u ** 2 - 2 * u ** 3), color=col, lw=0.75, ls=ls, zorder=2, alpha=al)
    c.text(2.68, 0.10, "listeners", color=INK2, va="center")
    if key:
        yk = h + 0.16
        for x, (kind, lab) in zip((0.30, 2.14, 3.86), key):
            col, ls, _ = styles[kind]
            c.plot([x, x + 0.30], [yk, yk], color=col, ls=ls, lw=0.9)
            c.text(x + 0.38, yk, lab, color=INK2, va="center")
    return c, h


def _lum(c):
    r = np.asarray(mpl.colors.to_rgb(c), float)
    r = np.where(r <= 0.04045, r / 12.92, ((r + 0.055) / 1.055) ** 2.4)
    return 0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2]


def contrast(a, b):
    la, lb = sorted([_lum(a), _lum(b)])
    return (lb + 0.05) / (la + 0.05)


def ink_on(fill):
    """White or ink text on a filled shape, whichever has the higher contrast (the journals ask for >= 4.5:1)."""
    return PAPER if contrast(PAPER, fill) >= contrast(INK, fill) else INK
