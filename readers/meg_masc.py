"""MEG-MASC release layout (listeners, sessions, stories, sound files, stimulus events) and listener trials."""
from __future__ import annotations

import ast
import json
import re
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from common.settings import dataset_root
from preprocessing.cnd_recordings import band_64

ROOT = Path(__file__).parents[1]
DS = dataset_root("meg_masc")
DER = ROOT / "data" / "derived" / "meg_masc"
TAB = ROOT / "results"
STORY = {0: "lw1", 1: "cable_spool_fort", 2: "easy_money", 3: "The_Black_Willow"}   # as in events.tsv (the README says "Widow")
FS, N_LAGS, WIN_S, EDGE_S = 64, 23, 0.8, 2.0


def seed_int(key: str) -> int:
    return zlib.crc32(f"9|{key}".encode()) % (2 ** 31)


def rng(key: str) -> np.random.Generator:
    return np.random.default_rng([9, zlib.crc32(key.encode())])


def bz(F: np.ndarray) -> np.ndarray:
    """1-8 Hz on the 64 Hz grid (band_64), then z-scored per row over the sound file."""
    B = band_64(np.atleast_2d(F), FS, 1.0, 8.0)
    return (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-12)


def norm_word(w) -> str:
    return re.sub(r"[^a-z0-9']", "", str(w).lower().replace("\u2019", "'"))


@lru_cache(None)
def participants() -> pd.DataFrame:
    return pd.read_csv(DS / "participants.tsv", sep="\t")


def sessions(sub: str) -> list[int]:
    return [s for s in (0, 1) if (DS / f"sub-{sub}" / f"ses-{s}" / "meg").exists()]


def listeners() -> list[str]:
    return [p.replace("sub-", "") for p in participants().participant_id]


def meg_dir(sub: str, ses: int) -> Path:
    return DS / f"sub-{sub}" / f"ses-{ses}" / "meg"


def events(sub: str, ses: int, task: int) -> pd.DataFrame:
    d = pd.read_csv(meg_dir(sub, ses) / f"sub-{sub}_ses-{ses}_task-{task}_events.tsv", sep="\t")
    tt = pd.DataFrame(list(d.trial_type.map(ast.literal_eval)))
    return pd.concat([d.drop(columns=["trial_type"]), tt], axis=1)


def sound_key(story: str, sound_id) -> str:
    return f"{story}_{int(float(sound_id))}"


def audio_path(key: str) -> Path:
    return DS / "stimuli" / "audio" / f"{key.lower()}.wav"


@lru_cache(None)
def stimulus_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(sounds, words, phonemes) of the stimulus, from one recording per story (timing within sound files is identical
    in every recording). Times are relative to each sound file's onset."""
    sounds, words, phones = [], [], []
    ref_sub = listeners()[0]
    for task, story in STORY.items():
        d = events(ref_sub, 0, task)
        for kind, out in (("word", words), ("phoneme", phones)):
            e = d[d.kind == kind].copy()
            e["story"], e["task"] = story, task
            e["key"] = [sound_key(story, s) for s in e.sound_id]
            e["t"] = e.start.astype(float)
            out.append(e)
        s = d[d.kind == "sound"].copy()
        s["story"], s["task"] = story, task
        s["key"] = [sound_key(story, x) for x in s.sound_id]
        sounds.append(s[["story", "task", "key", "sound_id"]])
    sounds = pd.concat(sounds, ignore_index=True)
    words = pd.concat(words, ignore_index=True)
    words["order"] = np.arange(len(words))
    words["w"] = words.word.map(norm_word)
    phones = pd.concat(phones, ignore_index=True)
    return sounds, words, phones


def sound_keys(story: str | None = None) -> list[str]:
    s = stimulus_tables()[0]
    s = s if story is None else s[s.story == story]
    return s.key.tolist()


def qc(sub: str, ses: int) -> dict:
    return json.loads((TAB / "qc" / f"sub-{sub}_ses-{ses}.json").read_text(encoding="utf-8"))


@lru_cache(None)
def inclusion() -> pd.DataFrame:
    return pd.read_csv(TAB / "meg_masc_inclusion.csv", dtype={"listener": str})


def usable_sessions(sub: str) -> list[int]:
    t = inclusion()
    return [int(s) for s in t[(t.listener == sub) & t.usable].session]


def two_session_listeners() -> list[str]:
    t = inclusion()
    return sorted(t[t.two_session_set].listener.unique())


def one_session_listeners() -> list[str]:
    t = inclusion()
    return sorted(t[t.one_session_set].listener.unique())


def features(key: str) -> dict:
    with np.load(DER / "features" / f"{key}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def kept_keys(sub: str, ses: int) -> list[str]:
    q = qc(sub, ses)
    return [k for st in q["stories"] if st["kept"] for k in st["sound_files"]]


def listener_trials(sub: str, ses: int, band: str = "d18") -> list[dict]:
    """Valid sound files of a usable session: {'key', 'story', 'Y' (channels x T, float64), 'v', 'T', 'F': {}, 'st'}."""
    out = []
    with np.load(DER / "meg" / f"sub-{sub}_ses-{ses}.npz") as z:
        for key in kept_keys(sub, ses):
            Y = z[f"{key}_{band}"].astype(np.float64)
            v = z[f"{key}_valid"]
            st = features(key)
            T = min(Y.shape[1], int(st["n"]))
            if v[:T].sum() < FS * 10:                        # fewer than 10 s of valid samples: not a fold
                continue
            out.append({"key": key, "story": key.rsplit("_", 1)[0], "Y": Y[:, :T], "v": v[:T].copy(), "T": T, "F": {}, "st": st})
    return out


def impulse(idx: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    R = np.zeros((1, n))
    idx = np.asarray(idx, int)
    ok = (idx >= 0) & (idx < n)
    np.add.at(R[0], idx[ok], np.asarray(values, float)[ok])
    return R
