"""Visual language of the main figures: colours, type sizes, canvases and shared marks."""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

Ramp = namedtuple("Ramp", "tint light mid base dark")
WORDS = Ramp("#ECF4FE", "#C9DBF0", "#739DCB", "#21609E", "#00396A")
SUNG = Ramp("#FFEFF2", "#F4CED5", "#D48294", "#A8385A", "#721736")
MELODY = Ramp("#FAF2E8", "#E8D6C0", "#D3AB76", "#B8812B", "#865A0E")
GLOBAL = Ramp("#F2F3F6", "#D6D9DE", "#8E959F", "#4C5562", "#29303A")
INK, INK2, INK3 = "#1E2329", "#565D66", "#8E959E"
RULE, BAND, PAPER = "#CDD1D6", "#F1EFEA", "#FFFFFF"
VERSION = {"speech": (WORDS, "spoken"), "song": (SUNG, "sung"), "melody": (MELODY, "hummed")}
W = 7.2                                         # printed width, inches (183 mm)
FS = {"letter": 8.5, "title": 7.0, "sub": 6.0, "label": 6.5, "tick": 6.0, "note": 6.0, "small": 5.6}
DZ = r"$\mathit{d}_\mathrm{z}$"
PV = r"$\mathit{p}$"


def p_txt(p, sub=None):
    """p as in the text (italic, lower case): one significant digit in scientific notation below 0.001, otherwise two
    significant digits."""
    lab = r"$\mathit{p}$" if not sub else r"$\mathit{p}_\mathrm{" + sub + "}$"
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        m = int(round(p / 10 ** e))
        if m == 10:
            m, e = 1, e + 1
        return f"{lab} = {m} × 10$^{{{e}}}$"
    if p < 0.01:
        return f"{lab} = {p:.4f}".rstrip("0")
    if p < 0.1:
        return f"{lab} = {p:.3f}"
    return f"{lab} = {p:.2f}"


def mean_ci(x, level=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x.mean(), stats.t.ppf(0.5 + level / 2, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)


def swarm_offsets(ax, values, size_pt=3.2, gap_pt=0.25, max_pt=None):
    """Deterministic beeswarm: horizontal offsets (points) that keep markers of diameter size_pt from overlapping,
    given the axes' current y-scale. Points are placed in order of value, each at the smallest free offset."""
    v = np.asarray(values, float)
    y0, y1 = ax.get_ylim()
    h_pt = ax.get_window_extent().height * 72 / ax.figure.dpi
    ypt = (v - y0) / (y1 - y0) * h_pt
    d = size_pt + gap_pt
    order = np.argsort(ypt)
    placed, off = [], np.zeros(v.size)
    for i in order:
        cands = [0.0] + [s * k * d * 0.5 for k in range(1, 400) for s in (1, -1)]
        for c in cands:
            if max_pt is not None and abs(c) > max_pt:
                continue
            if all((c - px) ** 2 + (ypt[i] - py) ** 2 >= d ** 2 for px, py in placed if abs(ypt[i] - py) < d):
                off[i] = c
                placed.append((c, ypt[i]))
                break
    return off


def pts_to_data_x(ax, pts):
    x0, x1 = ax.get_xlim()
    w_pt = ax.get_window_extent().width * 72 / ax.figure.dpi
    return np.asarray(pts) / w_pt * (x1 - x0)


def _text_polygon(t, r):
    """Display-space corners of a text's (possibly rotated) box."""
    rot = t.get_rotation() % 360
    bb = t.get_window_extent(r)
    if abs(rot) < 1e-6 or abs(rot - 360) < 1e-6:
        return np.array([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1]])
    t.set_rotation(0)
    b0 = t.get_window_extent(r)
    t.set_rotation(rot)
    w, h = b0.width, b0.height
    fx = {"left": 0.0, "center": 0.5, "right": 1.0}[t.get_ha()]
    fy = {"bottom": 0.0, "baseline": 0.0, "center": 0.5, "center_baseline": 0.5, "top": 1.0}[t.get_va()]
    loc = np.array([[-fx * w, -fy * h], [(1 - fx) * w, -fy * h], [(1 - fx) * w, (1 - fy) * h], [-fx * w, (1 - fy) * h]])
    a = np.radians(rot)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = loc @ R.T
    if t.get_rotation_mode() == "anchor":
        anchor = np.asarray(t.get_transform().transform(t.get_position()))
        return pts + anchor
    c = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
    return pts - pts.mean(0) + c


def _polys_overlap(p, q, tol=0.8):
    """Separating-axis test for two convex quadrilaterals (overlap deeper than tol pixels)."""
    for poly in (p, q):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]]) / (np.hypot(*e) + 1e-12)
            a, b = p @ n, q @ n
            if min(a.max(), b.max()) - max(a.min(), b.min()) <= tol:
                return False
    return True


def check_layout(fig, min_pt=5.0, ignore=()):
    """Report overlapping text (rotation-aware), text outside the figure and text smaller than min_pt (after a draw)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W_, H_ = fig.bbox.width, fig.bbox.height
    undrawn = set()                                   # tick labels outside the view interval are never drawn
    for a_ in fig.axes:
        for axis in (a_.xaxis, a_.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            eps = 1e-9 * max(1.0, abs(hi - lo))
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - eps <= tk.get_loc() <= hi + eps) or not a_.axison:
                    undrawn.update((id(tk.label1), id(tk.label2)))
    texts = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip() or t.get_alpha() == 0 or id(t) in undrawn:
            continue
        bb = t.get_window_extent(r)
        if bb.width < 0.5 or bb.height < 0.5:
            continue
        texts.append((t, bb, _text_polygon(t, r)))
    issues = []
    for t, bb, _ in texts:
        if t.get_fontsize() < min_pt - 1e-6:
            issues.append(f"small text {t.get_fontsize():.1f} pt: {t.get_text()[:40]!r}")
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W_ + 0.5 or bb.y1 > H_ + 0.5:
            issues.append(f"outside figure: {t.get_text()[:40]!r}")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            (t1, b1, p1), (t2, b2, p2) = texts[i], texts[j]
            if any(s in t1.get_text() or s in t2.get_text() for s in ignore):
                continue
            if min(b1.x1, b2.x1) - max(b1.x0, b2.x0) <= 0.8 or min(b1.y1, b2.y1) - max(b1.y0, b2.y0) <= 0.8:
                continue
            if _polys_overlap(p1, p2):
                issues.append(f"overlap: {t1.get_text()[:30]!r} / {t2.get_text()[:30]!r}")
    return issues
