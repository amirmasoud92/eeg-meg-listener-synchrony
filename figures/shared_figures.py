"""Saving, notation and the shared parts of the supplementary figures."""
from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import pandas as pd

from common.settings import PROJECT_ROOT
from figures import publication_panels as PF, supplementary_panels as SF
from figures.main_style import WORDS
from figures.numbering import FIG_MAP
from figures.supplementary_style import BAND, INK, INK2, INK3, LANG, RULE, SYNC, W, axes_in

FM = PROJECT_ROOT
OUT = PROJECT_ROOT / "figure_output"
T = PF.T
T7 = T
T9 = T
J = PF.J
_DZ, _P = re.compile(r"\bdz\b"), re.compile(r"\b[Pp](?= [=<])")


def save(fig, name, out=None):
    out = out or OUT
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=600, metadata={"Software": None})
    fig.savefig(out / f"{name}.pdf", dpi=600, metadata={"CreationDate": None, "Creator": None, "Producer": None})
    plt.close(fig)


def notation(s):
    """The statistical notation of the main text and figures (italic d_z and lower-case italic p) for a display string of the supplementary figure code, which writes 'dz' and 'P'; math segments ($...$) are left as they are."""
    if not isinstance(s, str) or ("dz" not in s and "P" not in s):
        return s
    parts = s.split("$")
    for i in range(0, len(parts), 2):                              # even parts lie outside math
        parts[i] = _P.sub(lambda m: r"$\mathit{p}$", _DZ.sub(lambda m: r"$\mathit{d}_\mathrm{z}$", parts[i]))
    return "$".join(parts)


@contextmanager
def si_notation():
    """Apply `notation` to every text drawn inside the block (titles, labels, annotations, tick labels)."""
    orig = mtext.Text.set_text
    mtext.Text.set_text = lambda self, s: orig(self, notation(s))
    try:
        yield
    finally:
        mtext.Text.set_text = orig


class _Through:
    """Route a module's `save(fig, name)` to `save` here, under the new name."""
    def __init__(self, module, names):
        self.module, self.names = module, names

    def __enter__(self):
        self.orig = self.module.save
        self.module.save = lambda fig, name: save(fig, self.names[name])

    def __exit__(self, *exc):
        self.module.save = self.orig


def renamed_supp(old: int):
    def draw():
        with _Through(SF, {f"FigS{old}": f"FigureS{FIG_MAP[old]}"}):
            getattr(SF, f"figS{old}")()
    return draw
