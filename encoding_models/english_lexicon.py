"""Word frequency (SUBTLEX-US, Zipf scale) and function-word list (Snowball) for the English stimuli."""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

LEX = Path(__file__).parents[1] / "data" / "lexicon"
PINS = {"subtlexus1.zip": "458128f90a28c4f396cb2a5b23ac93c56f745ee8cfca9be2afedad4091d15090",
        "subtlexus_zipf.csv": "883aae55f552f9b65e8d9729862c6c95af1ea6307a0794b72f7b413442c7484d",
        "snowball_english_stop.txt": "06b8b14bbe6f6e0b64c0830813819416d96a71bec1655668d83b3e602a651db6"}
ARCHAIC = {"o'er": "over"}
CLITICS = {"s", "t", "ve", "re", "ll", "d", "m"}


def check_pins():
    for f, h in PINS.items():
        got = hashlib.sha256((LEX / f).read_bytes()).hexdigest()
        if got != h:
            raise SystemExit(f"{f}: checksum {got} differs from the registered {h}")


@lru_cache(maxsize=1)
def _tables():
    check_pins()
    d = pd.read_csv(LEX / "subtlexus_zipf.csv", keep_default_na=False, na_values=[])
    d["w"] = d["Word"].astype(str).str.lower()
    zipf = d.groupby("w")["Zipf-value"].max().to_dict()
    k = ((d["FREQcount"] + 1) / 10 ** (d["Zipf-value"] - 3)).median()       # (corpus size + types) in millions
    oov = math.log10(1 / k) + 3
    stop = {ln.split("|")[0].strip() for ln in (LEX / "snowball_english_stop.txt").read_text(encoding="utf-8").splitlines()}
    stop.discard("")
    return zipf, oov, stop


def form(word: str) -> str:
    w = str(word).lower().replace("’", "'").replace("‘", "'").replace("", "'").strip(".,;:!?\"()")
    w = ARCHAIC.get(w, w)
    if w.endswith("in'"):
        w = w[:-1] + "g"
    return w


def _longer_part(w: str) -> str:
    """The host of an apostrophe form: before the apostrophe if a clitic follows, else the longer part."""
    a, _, b = w.partition("'")
    if b in CLITICS and a:
        return a
    return a if len(a) >= len(b) else b


def is_function(word: str) -> bool:
    _, _, stop = _tables()
    w = form(word)
    return w in stop or ("'" in w and _longer_part(w) in stop)


def zipf(word: str) -> tuple[float, bool]:
    """(Zipf value, in vocabulary)."""
    table, oov, _ = _tables()
    w = form(word)
    if any(c.isdigit() for c in w):
        return oov, False
    parts = [p for p in re.split(r"-+", w) if p]
    vals = []
    for p in parts:
        key = p if p in table else (_longer_part(p) if "'" in p else p)
        vals.append(table.get(key))
    if not vals or any(v is None for v in vals):
        known = [v for v in vals if v is not None]
        return (min(known) if known and len(parts) > 1 else oov), False
    return min(vals), True
