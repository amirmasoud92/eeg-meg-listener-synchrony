"""Acoustic regressors per sound file at 64 Hz for MEG-MASC (34 rows, as in the MEG attention dataset)."""
from __future__ import annotations

import math

import numpy as np
from scipy.io import wavfile

from encoding_models.meg_attention_regressors import acoustic
from readers import meg_masc as C

OUT = C.DER / "features"


def sound_file_regressors():
    for key in C.sound_keys():
        path = C.audio_path(key)
        fs, x = wavfile.read(path)
        n1000 = int(math.ceil(len(x) * 1000 / fs))
        n = int(math.ceil(n1000 * 8 / 125))
        np.savez(OUT / f"{key}.npz", n=np.array(n), n1000=np.array(n1000), acoustic_raw=acoustic(path, n))
