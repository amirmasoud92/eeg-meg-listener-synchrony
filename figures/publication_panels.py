"""Further shared panels and helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as ss
from matplotlib.patches import Rectangle
from scipy.io import wavfile

from common.settings import PROJECT_ROOT, dataset_root, load_paths

from figures.supplementary_style import BAND, INK, INK2, INK3, LANG, LANG_L, MUSIC, MUSIC_L, RULE, SONG, SONG_L, SYNC, SYNC_L, W, axes_in

ROOT = PROJECT_ROOT
OUT = PROJECT_ROOT / "figure_output"
STIM = dataset_root("cantisong") / "Stimuli"
P = load_paths()
T = PROJECT_ROOT / "figure_data"
TI, P5 = T, T
P6 = T


def J(path):
    return json.loads(Path(path).read_text())
