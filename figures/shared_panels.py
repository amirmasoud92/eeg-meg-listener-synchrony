"""Panels shared by the main figures, drawn from the result tables in figure_data/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as ss
from matplotlib.patches import Rectangle
from scipy.io import wavfile

from common.settings import PROJECT_ROOT, dataset_root, load_paths
from figures import brain as BR
from figures.main_style import BAND, FS, GLOBAL, INK, INK2, INK3, MELODY, PAPER, RULE, SUNG, WORDS

FM = PROJECT_ROOT
ROOT = PROJECT_ROOT
OUT = PROJECT_ROOT / "figure_output"
P = load_paths()
T = PROJECT_ROOT / "figure_data"
TI = TX = P5 = P6 = T7 = T8 = T78 = T9 = T
STIM = dataset_root("cantisong") / "Stimuli"
MEG_AUDIO = dataset_root("meg_attention") / "Audio"
_DK = {}
_BR_SURFACE = BR.surface                         # the original (cached) loader, captured before any patching


def J(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
