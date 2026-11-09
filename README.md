# Electrophysiological synchrony between listeners

Analysis code, figure code and figure data for the article *Neural synchrony reflects what listeners hear and who they are more than what they understand*.

Neural synchrony, the similarity of brain activity across people exposed to the same material, is widely used as a model-free index of shared processing. The article estimates what low-frequency synchrony (δ band, 1–4 Hz) between listeners reflects, using ten public EEG and MEG datasets in which the auditory input, the listener's state or the listener were varied while everything else was held fixed. This repository contains the implementation of the methods reported in the article, the code that draws its figures, the data plotted in those figures and the tables of the article.

The recordings themselves are not included. The analysis code is organised by method rather than as a pipeline to be run from start to finish.

## Publication

*Neural synchrony reflects what listeners hear and who they are more than what they understand.* Citation details will be added on publication.

## Datasets

All recordings are publicly available from their original sources and are not redistributed here.

| Dataset | Recording | Listeners | Contrast | Source |
| --- | --- | --- | --- | --- |
| Aalborg songs and speech (Simon et al., 2024) | EEG, 64 channels | 18 | Understood (Danish) and foreign (Finnish) speech and songs | [10.5281/zenodo.7500806](https://doi.org/10.5281/zenodo.7500806) |
| Etard and Reichenbach (2019) | EEG | 11 | English and Dutch stories in quiet and in babble | [10.5281/zenodo.7086209](https://doi.org/10.5281/zenodo.7086209) |
| Broderick et al. (2018), natural and time-reversed speech | EEG, 128 channels | 19 and 10 | Intact against time-reversed audiobook | [10.5061/dryad.070jc](https://doi.org/10.5061/dryad.070jc) |
| Speech in noise (Crosse et al., 2016) | EEG | 21 | Audio-only, video-only and audiovisual speech; intelligibility ratings | [10.5061/dryad.070jc](https://doi.org/10.5061/dryad.070jc) |
| Cocktail party (O'Sullivan et al., 2015) | EEG | 33 | Attended against ignored story; comprehension scores | [10.5061/dryad.070jc](https://doi.org/10.5061/dryad.070jc) |
| Alice (Brennan & Hale, 2019) | EEG, 61 channels | 48 | Comprehension across listeners | [10.7302/746w-g237](https://doi.org/10.7302/746w-g237) |
| CantiSong (Cantisani et al., 2023, 2024) | EEG, 64 channels | 20 | Spoken, sung and hummed versions of 18 songs | [10.5281/zenodo.14590637](https://doi.org/10.5281/zenodo.14590637) |
| Two audiobooks, one attended (Riegel et al., 2024; Schüller et al., 2023, 2025) | MEG, 248 magnetometers | 58 | Attended against ignored talker; comprehension scores | [10.5281/zenodo.12793944](https://doi.org/10.5281/zenodo.12793944), [12794769](https://doi.org/10.5281/zenodo.12794769), [12794808](https://doi.org/10.5281/zenodo.12794808) |
| MEG-MASC (Gwilliams et al., 2023) | MEG, 208 gradiometers | 27 (22 in two sessions) | Stability across days; sentences against word lists | [osf.io/rguwj](https://osf.io/rguwj/) |

## Repository layout

| Directory | Contents |
| --- | --- |
| `common/` | Loading of parameters and data locations; reproducible random streams |
| `config/` | Analysis parameters (`analysis_parameters.yaml`, `etard_parameters.yaml`) and the template for local data paths (`paths.yaml`) |
| `readers/` | Readers for the released file formats and dataset layouts |
| `preprocessing/` | EEG and MEG cleaning, epoch rejection and band-limiting |
| `synchrony/` | Correlated components analysis and the synchrony analysis of each dataset |
| `encoding_models/` | Stimulus regressors, language-model surprisal, transcription and forced alignment, ridge encoding models and the word-level gain |
| `decoding/` | Forward and backward models of selective attention |
| `network_states/` | Template source model, connectivity and spectral features, cross-validated representational geometry, VARX decomposition and the specification search |
| `statistical_tests/` | Permutation and sign-flip tests, effect-size intervals, equivalence tests, Bayes factors, multiplicity corrections and variance components |
| `figures/` | Code that draws Figures 1–6 and Supplementary Figures 1–5 |
| `figure_data/` | The values plotted in the figures (CSV and JSON) and the display data of the illustrative panels (`illustrations.npz`) |
| `table_data/` | Tables 1 and 2 and Supplementary Tables 1–4 of the article, as CSV files |

## Methods and their implementation

| Method | Module |
| --- | --- |
| Robust polynomial detrending, Hamming-window FIR filters, bad-channel interpolation, extended-Infomax ICA | `preprocessing/signal_operations.py`, `preprocessing/cnd_recordings.py` |
| FASTER and robust channel epoch rejection | `preprocessing/epoch_rejection.py` |
| MEG preprocessing: field interpolation, FastICA against synthetic ECG and proxy EOG, reference regression (MEG-MASC) | `preprocessing/meg_attention.py`, `preprocessing/meg_masc.py` |
| Correlated components analysis with shrinkage of the within-listener covariance; leave-one-out ISC; forward models | `synchrony/correlated_components.py` |
| Intact against time-reversed speech; the speech-defined space | `synchrony/intact_reversed_speech.py` |
| Understood against foreign speech and song | `synchrony/understood_foreign.py`, `synchrony/regional_synchrony.py` |
| Removal of the acoustic response by lagged ridge regression, with a mismatched-stimulus control | `synchrony/acoustic_removal.py`, `synchrony/cocktail_party_residuals.py`, `synchrony/alice_acoustic_removal.py` |
| Attention: same-story against other-story synchrony; forward and backward attention models | `synchrony/cocktail_party.py`, `decoding/attention_decoding.py` |
| Seeing the talker and rated intelligibility | `synchrony/speech_in_noise.py` |
| Comprehension, engagement and posterior α | `synchrony/alice_comprehension.py`, `synchrony/engagement_and_alpha.py`, `statistical_tests/meg_attention_tests.py` |
| Sung, hummed and spoken songs; simulation of the speech-specific excess | `synchrony/song_versions.py`, `synchrony/song_estimator_simulation.py` |
| Stability across days and generality across sounds | `synchrony/meg_masc.py`, `statistical_tests/meg_masc_tests.py`, `synchrony/generality_across_sounds.py` |
| Word-level gain: surprisal against permuted surprisal under a Monte Carlo rule | `encoding_models/word_level_gain.py`, `encoding_models/gpu_encoding.py`, `encoding_models/sufficient_statistics.py` |
| Surprisal (GPT-2, GerPT2), transcription and forced alignment | `encoding_models/surprisal.py`, `encoding_models/german_surprisal.py`, `encoding_models/transcription.py`, `encoding_models/word_alignment.py` |
| Template source model (fsaverage, eLORETA) and regional filters | `network_states/source_model.py`, `network_states/regional_filters.py` |
| dwPLI, orthogonalised amplitude-envelope correlation and spectral parameters | `network_states/connectivity_measures.py`, `network_states/spectral_parameters.py`, `network_states/network_features.py` |
| Crossnobis distances, language-capture index, alignment of comprehension axes, cross-decoding | `network_states/representational_geometry.py`, `network_states/language_capture.py` |
| VARX decomposition and Granger influences | `network_states/varx_model.py`, `network_states/varx_analysis.py`, `network_states/varx_tests.py` |
| Specification search with max-t correction; replication in the Etard data | `network_states/specification_search.py`, `network_states/specification_tests.py`, `network_states/etard_replication.py` |
| Sign-flip tests, noncentral-t intervals for d<sub>z</sub>, equivalence tests, JZS Bayes factors, Freedman-Lane partial correlations, Holm and Benjamini-Hochberg corrections, inverse-variance pooling | `statistical_tests/inference.py`, `statistical_tests/one_sided_bayes_factors.R` |
| Evidence for absence and intraclass correlations from crossed mixed models | `statistical_tests/evidence_for_absence.py` |

## Figures and tables

All figures of the article are drawn from the files in `figure_data/`, which hold the values the figures show: listener- and item-level estimates, test statistics, intervals and the display data of the illustrative panels. No recordings are needed. The cortical maps use the fsaverage template from MNE-Python:

```python
import mne
mne.datasets.fetch_fsaverage(subjects_dir="data/derived/mne_data")
```

Then, from the repository root:

```
python -m figures.make_figures            # all figures
python -m figures.make_figures 2 S4       # selected figures
```

The figures are written to `figure_output/` as PDF and PNG. Final typographic adjustments to the published figures were made in a vector editor.

The illustrative panels of Figures 1 and 6 (the CantiSong components) and the MEG sensor layout of Figure 4 come from `figure_data/illustrations.npz`, which `figures/illustration_data.py` computes from the preprocessed CantiSong recordings and the header of one MEG recording.

`table_data/` contains Tables 1 and 2 and Supplementary Tables 1–4 as they appear in the article.

## Requirements

- Python 3.11 with the packages listed in `requirements.txt` (NumPy, SciPy, pandas, scikit-learn, statsmodels, MNE-Python, specparam, PyTorch, Transformers, Matplotlib).
- R 4.4.2 with lme4 1.1-37, lmerTest 3.1-3, BayesFactor 0.9.12-4.7 and jsonlite, for the mixed models and Bayes factors.
- Montreal Forced Aligner 3.4.2 with the pretrained German acoustic model, dictionary and grapheme-to-phoneme model, for the word timings of the MEG attention dataset.

The word-level encoding models run in double precision on a GPU when one is available; set `ENCODING_DEVICE` (for example `cuda:0` or `cpu`) to choose the device.

```
pip install -r requirements.txt
```

## Configuration

Set the locations of the downloaded releases in `config/paths.yaml`. Paths are relative to the repository root. Modules are imported from the repository root, for example:

```python
from synchrony import correlated_components
from statistical_tests.inference import sign_flip_test, jzs_bf10
```

## Example

Correlated components analysis on simulated data: twelve listeners share one source mixed into 32 channels, with independent noise.

```python
import numpy as np

from synchrony.correlated_components import _cov_terms, corrca_from_terms, subject_isc

rng = np.random.default_rng(0)
source = rng.standard_normal((1, 6000))
mixing = rng.standard_normal((32, 1))
data = np.stack([mixing @ source + 2 * rng.standard_normal((32, 6000)) for _ in range(12)])

Rw, Rb = _cov_terms(data)                       # within- and between-listener covariance
W, _ = corrca_from_terms(Rw, Rb, len(data), shrinkage=0.5, n_components=3)
isc = subject_isc(data, W)                      # listeners x components
print(isc[:, 0].round(2))
```

In the analyses, the projections are fitted on the other stimuli (runs, songs, chapters or sound files) and applied to the held-out one, so that no stimulus is scored with a projection fitted to it.

## Reproducibility

Random draws come from named streams (`common.settings.rng`) derived from the seeds in the configuration files. Seed values and stream keys only label the streams; changing them changes the draws.

Some analyses read intermediate tables that are produced from the public data by the code here, such as preprocessed recordings, word tables and encoding-model outputs. The transcripts made for the MEG attention dataset are not included, because the audiobook texts are under copyright. The word-frequency lexicons (SUBTLEX-US, SUBTLEX-DE) and the Snowball stop-word lists must be obtained from their providers and placed in `data/lexicon/`.

The VARX model is a Python port of the MATLAB toolbox by Parra et al. (2025), [github.com/lcparra/varx](https://github.com/lcparra/varx).

## Licence

The code and the files in `figure_data/` and `table_data/` are released under the MIT Licence (see `LICENSE`). The values in `figure_data/` are derived from the public datasets listed above, which remain under the licences of their original releases.

## Contact

Amirmasoud Ahmadi
Max Planck Institute for Biological Intelligence
Amirmasoud.Ahmadi@bi.mpg.de
