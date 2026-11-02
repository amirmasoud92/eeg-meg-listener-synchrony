"""Template source model (fsaverage), eLORETA kernel and regional spatial filters for 18 Desikan-Killiany regions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np

REGIONS = (
    ("TTG", ("transversetemporal",)),
    ("STG", ("superiortemporal",)),
    ("MTG", ("middletemporal",)),
    ("IFG", ("parsopercularis", "parstriangularis", "parsorbitalis")),
    ("PreCG", ("precentral",)),
    ("SMG", ("supramarginal",)),
    ("IPL", ("inferiorparietal",)),
    ("RMFG", ("rostralmiddlefrontal",)),
    ("PCun", ("precuneus",)),
)
HEMIS = ("lh", "rh")


def region_names() -> list[str]:
    return [f"{name}_{h}" for h in HEMIS for name, _ in REGIONS]


def make_info(ch_names: list[str], sfreq: float) -> mne.Info:
    info = mne.create_info(ch_names, sfreq, "eeg")
    try:
        montage = mne.channels.make_standard_montage("colin27_1005")
    except ValueError:
        montage = mne.channels.make_standard_montage("standard_1005")
    info.set_montage(montage, match_case=False)
    with mne.utils.use_log_level("error"):
        tmp = mne.io.RawArray(np.zeros((len(ch_names), 2)), info)
        tmp.set_eeg_reference("average", projection=True)
    return tmp.info


def load_labels(subjects_dir: Path) -> list[mne.Label]:
    """18 regions (9 per hemisphere) from the Desikan-Killiany atlas; IFG merges the three pars."""
    aparc = {lab.name: lab for lab in mne.read_labels_from_annot("fsaverage", "aparc", subjects_dir=subjects_dir,
                                                                 verbose=False)}
    labels = []
    for h in HEMIS:
        for name, parts in REGIONS:
            lab = aparc[f"{parts[0]}-{h}"]
            for p in parts[1:]:
                lab = lab + aparc[f"{p}-{h}"]
            lab.name = f"{name}_{h}"
            labels.append(lab)
    return labels


@dataclass
class TemplateModel:
    info: mne.Info
    fwd: mne.Forward
    src: mne.SourceSpaces
    labels: list[mne.Label]
    label_vertices: list[np.ndarray] = field(repr=False)
    label_flip: list[np.ndarray] = field(repr=False)

    @classmethod
    def build(cls, ch_names: list[str], sfreq: float, subjects_dir: Path, cache: Path,
              mindist_mm: float = 5.0) -> "TemplateModel":
        info = make_info(ch_names, sfreq)
        fs_dir = subjects_dir / "fsaverage"
        fwd_path = cache / "fsaverage-ico5-eeg64-fwd.fif"
        if fwd_path.exists():
            fwd = mne.read_forward_solution(fwd_path, verbose=False)
        else:
            src = mne.read_source_spaces(fs_dir / "bem" / "fsaverage-ico-5-src.fif", verbose=False)
            fwd = mne.make_forward_solution(info, trans="fsaverage", src=src,
                                            bem=fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif",
                                            eeg=True, meg=False, mindist=mindist_mm, n_jobs=8, verbose=False)
            mne.write_forward_solution(fwd_path, fwd, overwrite=True, verbose=False)
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, verbose=False)
        src = fwd["src"]
        labels = load_labels(subjects_dir)
        offsets = [0, len(src[0]["vertno"])]
        verts, flips = [], []
        for lab in labels:
            hemi = 0 if lab.hemi == "lh" else 1
            vno = src[hemi]["vertno"]
            idx = np.searchsorted(vno, np.intersect1d(vno, lab.vertices))
            verts.append(idx + offsets[hemi])
            flips.append(mne.label_sign_flip(lab, src))
        return cls(info=info, fwd=fwd, src=src, labels=labels, label_vertices=verts, label_flip=flips)

    # -- inverse kernels ---------------------------------------------------------------
    def _identity_evoked(self) -> mne.EvokedArray:
        return mne.EvokedArray(np.eye(len(self.info.ch_names)), self.info, tmin=0.0, nave=1, verbose=False)

    def eloreta_kernel(self, lambda2: float, loose: float, depth: float | None) -> np.ndarray:
        """Source-by-channel eLORETA kernel (normal orientation), including the average-reference projector."""
        noise_cov = mne.make_ad_hoc_cov(self.info, verbose=False)
        inv = mne.minimum_norm.make_inverse_operator(self.info, self.fwd, noise_cov, loose=loose, depth=depth,
                                                     verbose=False)
        stc = mne.minimum_norm.apply_inverse(self._identity_evoked(), inv, lambda2=lambda2, method="eLORETA",
                                             pick_ori="normal", verbose=False)
        return stc.data

    def lcmv_kernel(self, data_cov: np.ndarray, n_samples: int, reg: float, rank: int,
                    weight_norm: str = "unit-noise-gain", pick_ori: str = "max-power") -> np.ndarray:
        cov = mne.Covariance(data_cov * 1e-12, self.info.ch_names, bads=[], projs=self.info["projs"],
                             nfree=n_samples - 1, verbose=False)
        filt = mne.beamformer.make_lcmv(self.info, self.fwd, cov, reg=reg, noise_cov=None, pick_ori=pick_ori,
                                        weight_norm=weight_norm, rank={"eeg": rank}, verbose=False)
        stc = mne.beamformer.apply_lcmv(self._identity_evoked(), filt, verbose=False)
        return stc.data

    # -- regional filters --------------------------------------------------------------
    def regional_filters(self, kernel: np.ndarray, data_cov: np.ndarray) -> np.ndarray:
        """Spatial filters (n_regions x n_channels) equivalent to MNE 'pca_flip' on the pooled data."""
        W = np.zeros((len(self.labels), kernel.shape[1]))
        for r, (idx, flip) in enumerate(zip(self.label_vertices, self.label_flip)):
            K = kernel[idx]
            M = K @ data_cov @ K.T
            ev, vec = np.linalg.eigh(M)
            u, lam1 = vec[:, -1], ev[-1]
            sign = np.sign(u @ flip) or 1.0
            scale = np.sqrt(max(np.trace(M), 0) / lam1) / np.sqrt(len(idx))
            W[r] = sign * scale * (u @ K)
        return W

    def label_leadfields(self) -> np.ndarray:
        """Channel-by-region leadfield of spatially uniform, sign-aligned unit activity in each label."""
        G = self.fwd["sol"]["data"]
        n_src = G.shape[1] // 3
        normal = G.reshape(G.shape[0], n_src, 3)[:, :, 2]
        normal = normal - normal.mean(axis=0, keepdims=True)
        return np.column_stack([normal[:, idx] @ flip for idx, flip in zip(self.label_vertices, self.label_flip)])
