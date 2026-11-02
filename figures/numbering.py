"""Figure numbering."""
from __future__ import annotations

import re

FIG_MAP = {1: 1, 13: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12, 12: 13, 11: 14}
TAB_MAP = {1: 1, 2: 2, 3: 3, 9: 4, 7: 5, 8: 6, 4: 7, 5: 8, 10: 9, 6: 10, 11: 11}
_REF = re.compile(r"\b(Figs?\.|Figures?|Tables?)(\s+)(S\d+[a-z]?(?:(?:\s*(?:,|–|and)\s*)S\d+[a-z]?)*)")
_ITEM = re.compile(r"S(\d+)–S(\d+)|S(\d+)([a-z]?)")
