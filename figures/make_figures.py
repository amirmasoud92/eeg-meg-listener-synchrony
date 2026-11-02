"""Draw the figures of the article from the tables in figure_data/.

    python -m figures.make_figures            # every figure
    python -m figures.make_figures 4 S2       # selected figures
Output: figure_output/ (PDF and PNG).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from common.settings import PROJECT_ROOT
from figures import design as V, main_figures as MF, panels as PL, shared_figures as PF, supplementary_panels as SF, supplementary_style as S
from figures.main_style import FS, INK2

ROOT = PROJECT_ROOT
OUT = PROJECT_ROOT / "figure_output"
RELABEL = {}
SAVE_SI = lambda fig, name: PF.save(fig, name, OUT)
MAIN = {k: getattr(MF, f"figure{k}") for k in "123456"}


def _relabel(fig, pairs):
    """Replace label substrings in every text of `fig`: figure and axes texts, and tick labels (reset through the axis,
    since matplotlib regenerates tick text at draw time). Every pair must match. Figure and axes text is set directly, so
    the notation already applied by si_notation is not applied twice."""
    import matplotlib.text as mtext
    hits = {a: 0 for a, _ in pairs}

    def sub(s):
        for a, b in pairs:
            if a in s:
                hits[a] += 1
                s = s.replace(a, b)
        return s
    ticks = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labs = axis.get_ticklabels()
            ticks |= {id(t) for t in labs}
            old = [t.get_text() for t in labs]
            if old and any(sub(x) != x for x in old):
                axis.set_ticks(axis.get_ticklocs(), [sub(x) for x in old], fontsize=labs[0].get_fontsize())
    for t in fig.findobj(mtext.Text):
        if id(t) in ticks:
            continue
        s = t.get_text()
        new = sub(s)
        if new != s:
            t._text, t.stale = new, True
    assert all(hits.values()), hits


def _second_line(fig, first: str, line: str, dy: float = 0.13):
    """Put `line` under the figure text that starts with `first` (same position, style and alignment), dy inches lower."""
    t = next(t for t in fig.texts if t.get_text().startswith(first))
    x, y = t.get_position()
    fig.text(x, y - dy / fig.get_size_inches()[1], line, fontsize=t.get_fontsize(), color=t.get_color(), ha=t.get_ha(),
             va=t.get_va())


def _relabel_part(fig, old: int):
    if old in RELABEL:
        _relabel(fig, RELABEL[old])
    if old == 4:                      # panel a: the longer subtitle on two lines, clear of panel b's subtitle
        _relabel(fig, [("listener means over runs; ", "")])
        _second_line(fig, "same − other story over 30 runs", "listener means over runs")


def _relabelled(old: int, new: str):
    """A supplementary figure (supplementary_panels.figS<old>) saved under its number in the article, relabelled (RELABEL)."""
    def draw():
        orig = SF.save
        SF.save = lambda fig, name: (_relabel_part(fig, old), PF.save(fig, new, OUT))
        try:
            getattr(SF, f"figS{old}")()
        finally:
            SF.save = orig
    return draw


def _network_study():
    """Supplementary Fig. 2: the network-state study with the one-sided Bayes factor of the global test (H1)
    written above its index in panel b, which merges the global test's own figure into this one."""
    bf = PF.J(PF.T / "aalborg_network_tests.json")["bayes"]["jzs"]["H1"]["bf10_positive"]

    def save(fig, name):
        ax = next(a for a in fig.axes if "primary" in [t.get_text() for t in a.get_xticklabels()])
        ax.text(0.2, 1.10, f"H1: BF$_{{+0}}$ {bf:.2f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=5.4, color=INK2)
        PF.save(fig, "FigureS2", OUT)
    orig = SF.save
    SF.save = save
    try:
        SF.figS2()
    finally:
        SF.save = orig


def _stacked(parts, new: str):
    """Supplementary figures drawn with relabelled panel letters and stacked on one page (PDF, vector, and PNG). parts: [(figure number, {old letter: new letter})]."""
    def draw():
        import pypdf
        from PIL import Image
        tmp = OUT / "_parts"
        names = []
        for old, mapping in parts:
            orig_save, orig_head = SF.save, S.fig_heading
            SF.save = lambda fig, name, o=old: (_relabel_part(fig, o), PF.save(fig, f"{new}_part{o}", tmp))
            S.fig_heading = lambda fig, x, y, letter, title, sub=None, _m=mapping, **kw: orig_head(fig, x, y, _m.get(letter, letter), title, sub, **kw)
            try:
                getattr(SF, f"figS{old}")()
            finally:
                SF.save, S.fig_heading = orig_save, orig_head
            names.append(tmp / f"{new}_part{old}")
        pages = [pypdf.PdfReader(str(n) + ".pdf").pages[0] for n in names]
        width = max(float(pg.mediabox.width) for pg in pages)
        height = sum(float(pg.mediabox.height) for pg in pages)
        writer = pypdf.PdfWriter()
        page = writer.add_blank_page(width=width, height=height)
        y = height
        for pg in pages:
            y -= float(pg.mediabox.height)
            page.merge_transformed_page(pg, pypdf.Transformation().translate(0, y))
        with open(OUT / f"{new}.pdf", "wb") as f:
            writer.write(f)
        ims = [Image.open(str(n) + ".png").convert("RGB") for n in names]
        out = Image.new("RGB", (max(i.width for i in ims), sum(i.height for i in ims)), "white")
        y = 0
        for i in ims:
            out.paste(i, (0, y))
            y += i.height
        out.save(OUT / f"{new}.png", dpi=(600, 600))
    return draw


def main(which=None):
    which = which or list(MAIN) + list(SUPP)
    base = mpl.rcParams.copy()
    for w in which:
        mpl.rcParams.update(base)                                  # the same starting style for every figure
        if w in MAIN:
            V.setup()
            MAIN[w]()
        else:
            S.setup()
            with PF.si_notation():
                SUPP[w]()
        plt.close("all")


SUPP = {"S1": lambda: PL.figS1_marked("synchrony", SAVE_SI), "S2": _network_study, "S3": _relabelled(3, "FigureS3"),
        "S4": _stacked([(4, {}), (5, {"a": "d", "b": "e"})], "FigureS4"), "S5": lambda: PL.figS_further(SAVE_SI)}
