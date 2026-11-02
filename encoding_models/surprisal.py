"""Word surprisal under a GPT-2 model: the negative log probability of each word given its preceding text."""
from __future__ import annotations

import numpy as np
import torch

def surprisal_long(text: str, spans: list[tuple[int, int]], tok, model, win: int = 1024, stride: int = 512) -> list[float]:
    """Per-span surprisal (bits) with sliding windows: each token is scored in the window that gives it the most
    context (windows of `win` tokens advancing by `stride`; the first token is conditioned on <|endoftext|>)."""
    enc = tok(text, return_offsets_mapping=True)
    ids = [tok.eos_token_id] + enc["input_ids"]
    offs = [(0, 0)] + enc["offset_mapping"]
    tok_s = np.full(len(ids), np.nan)
    start = 0
    while True:
        end = min(start + win, len(ids))
        with torch.no_grad():
            logp = torch.log_softmax(model(torch.tensor([ids[start:end]])).logits[0], dim=-1)
        first = 1 if start == 0 else start + (win - stride)
        for k in range(max(first, start + 1), end):
            tok_s[k] = float(-logp[k - start - 1, ids[k]] / np.log(2))
        if end == len(ids):
            break
        start += stride
    out = []
    for a, b in spans:
        sel = [v for (o0, o1), v in zip(offs, tok_s) if o1 > a and o0 < b]
        out.append(float(np.nansum(sel)) if sel else np.nan)
    return out
