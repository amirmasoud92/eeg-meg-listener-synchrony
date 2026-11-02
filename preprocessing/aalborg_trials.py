"""Aalborg data: clean trials, epoch masks and the rule for including trials and listeners."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common.settings import AnalysisConfig, load_analysis, load_paths
from preprocessing.epoch_rejection import reject
from preprocessing.epochs import Trial, coarse_mask, load_participant
from readers.aalborg_release import load_channel_names
from types import SimpleNamespace

@dataclass
class CleanTrial:
    trial: Trial
    ok_2s: np.ndarray = field(repr=False)
    n_equalised_2s: int
    n_equalised_4s: int
    pad: int = 0

    @property
    def ok_4s(self) -> np.ndarray:
        k = self.ok_2s.size // 2
        return self.ok_2s[: 2 * k].reshape(k, 2).all(axis=1)

    def sample_mask(self, epoch_s: float = 2.0) -> np.ndarray:
        """Boolean per-sample validity on the trial's time axis (trailing partial epoch invalid)."""
        n = int(round(epoch_s * self.trial.sfreq))
        mask = np.zeros(self.trial.data.shape[1], dtype=bool)
        mask[self.pad: self.pad + self.ok_2s.size * n] = np.repeat(self.ok_2s, n)
        return mask


class CleanDataset:
    def __init__(self, paths: SimpleNamespace, cfg: AnalysisConfig):
        self.paths, self.cfg = paths, cfg
        self.table = pd.read_csv(paths.processed_root / "trial_table.csv")
        self.inclusion = pd.read_csv(paths.derived_root / "epochs" / "trial_inclusion.csv").set_index("key")
        self.masks = dict(np.load(paths.derived_root / "epochs" / "epoch_masks_2s.npz"))
        s2 = cfg.section("preprocessing")
        self.window = tuple(s2["analysis_window_s"])
        self.epoch_s = s2["connectivity_epoch_s"]

    @property
    def subjects(self) -> list[int]:
        inc = self.inclusion[self.inclusion.included]
        return sorted(inc.subject.unique().tolist())

    def participant(self, subject: int, pad_s: float = 0.0) -> list[CleanTrial]:
        """Included trials; ``pad_s`` seconds preceding the analysis window are kept for filtering."""
        if pad_s > self.window[0]:
            raise ValueError("padding extends before trial onset")
        sfreq = self.cfg.dataset.preprocessed_sfreq
        trials = load_participant(self.paths.preprocessed_dir, self.table, subject,
                                  (self.window[0] - pad_s, self.window[1]), sfreq)
        out = []
        for t in trials:
            row = self.inclusion.loc[t.key]
            if not row.included:
                continue
            out.append(CleanTrial(trial=t, ok_2s=self.masks[t.key].astype(bool),
                                  n_equalised_2s=int(row.equalised_n_2s), n_equalised_4s=int(row.equalised_n_4s),
                                  pad=int(round(pad_s * sfreq))))
        return out

    def pooled_covariance(self, trials: list[CleanTrial]) -> tuple[np.ndarray, int]:
        """Channel covariance over all clean samples of the given trials (pooled over conditions)."""
        cov, n = 0.0, 0
        for ct in trials:
            x = ct.trial.data[:, ct.sample_mask(self.epoch_s)]
            x = x - x.mean(axis=1, keepdims=True)
            cov = cov + x @ x.T
            n += x.shape[1]
        return cov / (n - 1), n


def reject_listener_epochs(subject, paths, table, cfg2, ch_names, sfreq, seed):
    trials = load_participant(paths.preprocessed_dir, table, subject, tuple(cfg2["analysis_window_s"]), sfreq)
    res = reject(trials, ch_names, cfg2["connectivity_epoch_s"], seed=seed + subject,
                 max_bad_fraction=cfg2["max_bad_epoch_fraction"], method=cfg2["epoch_rejection"],
                 z_threshold=cfg2["faster_z"], channel_robust_z=cfg2["channel_robust_z"])
    factor = int(round(cfg2["spectral_epoch_s"] / cfg2["connectivity_epoch_s"]))
    rows, masks = [], {}
    for t in trials:
        ok2 = res.epoch_ok[t.key]
        ok4 = coarse_mask(ok2, factor)
        masks[t.key] = ok2
        rows.append({"subject": subject, "condition": t.condition, "excerpt": t.excerpt, "label": t.label,
                     "key": t.key, "presentation_index": t.presentation_index,
                     "n_epochs_2s": ok2.size, "n_clean_2s": int(ok2.sum()), "n_clean_4s": int(ok4.sum()),
                     "trial_retained": res.trial_ok[t.key]})
    return res.threshold_uv, pd.DataFrame(rows), masks


def trial_inclusion():
    paths, cfg = load_paths(), load_analysis()
    cfg2 = cfg.section("preprocessing")
    sfreq = cfg.dataset.preprocessed_sfreq
    out_d = paths.derived_root / "epochs"
    out_q = paths.results_root / "qc"
    ch_names = load_channel_names(paths.channel_list)
    table = pd.read_csv(paths.processed_root / "trial_table.csv")

    results = Parallel(n_jobs=6)(delayed(reject_listener_epochs)(s, paths, table, cfg2, ch_names, sfreq, cfg.project.seed)
                                 for s in cfg.dataset.subjects)
    thresholds = {s: r[0] for s, r in zip(cfg.dataset.subjects, results)}
    trials = pd.concat([r[1] for r in results], ignore_index=True)
    masks = {k: v for r in results for k, v in r[2].items()}
    np.savez_compressed(out_d / "epoch_masks_2s.npz", **masks)

    kept = trials[trials.trial_retained]
    counts = kept.pivot_table(index="subject", columns="condition", values="key", aggfunc="count").reindex(
        columns=cfg.dataset.conditions).fillna(0).astype(int)
    participant_ok = (counts >= cfg2["min_trials_per_condition"]).all(axis=1)
    trials["participant_retained"] = trials.subject.map(participant_ok)
    trials["included"] = trials.trial_retained & trials.participant_retained
    n_min = trials[trials.included].groupby("subject").n_clean_2s.min()
    n_min4 = trials[trials.included].groupby("subject").n_clean_4s.min()
    trials["equalised_n_2s"] = trials.subject.map(n_min)
    trials["equalised_n_4s"] = trials.subject.map(n_min4)
    trials.to_csv(out_d / "trial_inclusion.csv", index=False)
    counts.to_csv(out_q / "retained_trials.csv")
    pd.Series(thresholds, name="threshold_uv").to_csv(out_q / "thresholds.csv")
