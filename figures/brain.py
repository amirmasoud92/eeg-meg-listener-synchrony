"""Cortical surface rendering of regional values on the inflated fsaverage surface."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from network_states.source_model import REGIONS

VIEWS = {
    "left": (np.array([0, -1, 0]), np.array([0, 0, 1]), np.array([-1, 0, 0])),
    "right": (np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1, 0, 0])),
    "left_medial": (np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1, 0, 0])),
    "right_medial": (np.array([0, -1, 0]), np.array([0, 0, 1]), np.array([-1, 0, 0])),
    "dorsal": (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])),
}
HEMI = {"left": ("lh",), "right": ("rh",), "left_medial": ("lh",), "right_medial": ("rh",), "dorsal": ("lh", "rh")}
DK_TO_ROI = {dk: roi for roi, dks in REGIONS for dk in dks}
GYRUS, SULCUS = "#EDEBE6", "#CFCCC5"


@lru_cache(maxsize=8)
def surface(fs_dir: str, hemi: str, surf: str, full: bool = True):
    """Vertices, triangles (full resolution, or the ico-5 subset), analysis-region name per vertex ('' outside) and sulcal depth, for one hemisphere."""
    import mne
    import nibabel as nib
    fs = Path(fs_dir)
    src = mne.read_source_spaces(fs / "bem" / "fsaverage-ico-5-src.fif", verbose=False)[0 if hemi == "lh" else 1]
    rr, faces = nib.freesurfer.read_geometry(fs / "surf" / f"{hemi}.{surf}")
    labels, _, names = nib.freesurfer.read_annot(fs / "label" / f"{hemi}.aparc.annot")
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    roi = np.array([f"{DK_TO_ROI[names[k]]}_{hemi}" if k >= 0 and names[k] in DK_TO_ROI else "" for k in labels], dtype=object)
    sulc = nib.freesurfer.read_morph_data(fs / "surf" / f"{hemi}.sulc")
    rr = rr.astype(float)
    if surf != "pial" and hemi in ("lh", "rh"):          # separate inflated hemispheres for a dorsal view
        rr[:, 0] += (-1 if hemi == "lh" else 1) * (rr[:, 0].max() - rr[:, 0].min()) * 0.06
    return rr, (np.asarray(faces) if full else np.asarray(src["use_tris"])), roi, sulc


def draw(ax, values: dict, fs_dir, view: str = "left", cmap=None, norm=None, surf: str = "inflated",
         faded: set | None = None, fade: float = 0.62, outline: str | None = None, outline_lw: float = 0.35,
         light=(0.35, 0.55, 1.0), ambient: float = 0.52, highlight: dict | None = None, full: bool = True) -> None:
    """Draw region ``values`` ('TTG_rh' -> value) on ``ax``. Regions in ``faded`` are blended towards the
    background (e.g. non-significant). ``highlight`` maps region -> colour for categorical fills. The surface is
    rasterised (vector text and axes stay vector in PDF)."""
    cmap = cmap or mpl.colormaps["viridis"]
    norm = norm or mpl.colors.Normalize(0, 1)
    faded = faded or set()
    highlight = highlight or {}
    sx, sy, sz = VIEWS[view]
    L = np.asarray(light, float) / np.linalg.norm(light)
    polys, cols, depth, edges = [], [], [], []
    g_rgb, s_rgb = np.array(mpl.colors.to_rgb(GYRUS)), np.array(mpl.colors.to_rgb(SULCUS))
    for hemi in HEMI[view]:
        rr, tris, roi, sulc = surface(str(fs_dir), hemi, surf, full)
        P = np.c_[rr @ sx, rr @ sy, rr @ sz]
        T = P[tris]
        nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        front = nrm[:, 2] > 0
        fr = roi[tris]                                            # region of each corner (majority vote per face)
        face_roi = np.where((fr[:, 0] == fr[:, 1]) | (fr[:, 0] == fr[:, 2]), fr[:, 0], fr[:, 1])
        s = (sulc[tris].mean(1) > 0).astype(float)[:, None]       # sulcal depth > 0: sulcus
        base = g_rgb * (1 - s) + s_rgb * s
        for r in set(face_roi) - {""}:
            if r not in values and r not in highlight:
                continue
            m = face_roi == r
            c = np.array(mpl.colors.to_rgb(highlight[r])) if r in highlight else np.array(cmap(norm(values[r]))[:3])
            cc = c[None, :] * (1 - fade) + base[m] * fade if r in faded else np.repeat(c[None, :], m.sum(), 0)
            base[m] = cc * (0.93 + 0.07 * (1 - s[m]))              # keep a trace of the sulcal pattern inside regions
        shade = ambient + (1 - ambient) * np.clip(nrm @ L, 0, 1)
        rgb = np.clip(base * shade[:, None], 0, 1)
        polys.append(T[front][:, :, :2])
        cols.append(rgb[front])
        depth.append(T[front][:, :, 2].mean(1))
        if outline:                                               # region borders on visible faces
            tf = tris[front]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                ra, rb = roi[tf[:, a]], roi[tf[:, b]]
                m = (ra != rb) & ((ra != "") | (rb != ""))
                edges.append(np.stack([P[tf[m, a]][:, :2], P[tf[m, b]][:, :2]], 1))
    polys, cols, depth = np.concatenate(polys), np.concatenate(cols), np.concatenate(depth)
    order = np.argsort(depth)
    pc = PolyCollection(polys[order], facecolors=cols[order], edgecolors=cols[order], linewidths=0.1, zorder=1)
    pc.set_rasterized(True)
    ax.add_collection(pc)
    if outline and edges:
        lc = LineCollection(np.concatenate(edges), colors=outline, linewidths=outline_lw, zorder=2, capstyle="round")
        lc.set_rasterized(True)
        ax.add_collection(lc)
    xy = polys.reshape(-1, 2)
    pad = 2.0
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def centroids(fs_dir, surf: str = "inflated", view: str = "left") -> dict:
    """Projected 2-D centroid of each analysis region for ``view`` (for labels or network nodes)."""
    sx, sy, sz = VIEWS[view]
    out = {}
    for hemi in HEMI[view]:
        rr, _, roi, _ = surface(str(fs_dir), hemi, surf)
        for r in set(roi) - {""}:
            p = rr[roi == r].mean(0)
            out[r] = (p @ sx, p @ sy, p @ sz)
    return out
