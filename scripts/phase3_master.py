#!/usr/bin/env python3
"""
Phase III robustness and sensitivity benchmark for cross-session MI-BCI backdoor defense.

Purpose
-------
Phase II showed that calibration-referenced class-consistency screening (CRCS) can
strongly reduce targeted backdoor attack success while largely preserving clean
cross-session decoding. Phase III is a pre-specified robustness study designed to
answer the main reviewer-critical question: does that result persist when the
poisoning prevalence and the maximum removal budget are varied independently?

Dataset
-------
BCI Competition IV Dataset 2a (MOABB: BNCI2014_001): 9 subjects, 2 sessions,
22 EEG channels, 250 Hz, 4 motor-imagery classes.

Primary design
--------------
- Strict inductive Session-1 -> Session-2 evaluation only.
- Target calibration: 25%, 50%, 80%.
- Poisoning rates: 5%, 10%, 15%, 20% of source-session training trials.
- CRCS removal-budget caps: 5%, 10%, 15%, 20%, 25%.
- 9 subjects x 5 seeds.
- Primary pre-specified CRCS budget: 15% (carried forward from Phase II).
- Primary inferential family: poisoned vs CRCS ASR at the 15% budget across
  3 calibration levels x 4 poison rates = 12 paired subject-level tests,
  Holm-corrected within that family.

Scientific status
-----------------
CRCS remains a candidate method under validation. This script does not hard-code
performance values and does not assume the defense succeeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import warnings
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid
from scipy.stats import rankdata, wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

try:
    import mne
    from mne.decoding import CSP
    mne.set_log_level("WARNING")
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "MNE is required. Install the tested environment with:\n"
        "  python -m pip install -r requirements_colab.txt"
    ) from exc


# -----------------------------------------------------------------------------
# Reproducibility and configuration
# -----------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


@dataclass(frozen=True)
class Config:
    subjects: Tuple[int, ...] = tuple(range(1, 10))
    seeds: Tuple[int, ...] = (11, 22, 33, 44, 55)
    calibration_fractions: Tuple[float, ...] = (0.25, 0.50, 0.80)
    poison_rates: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
    removal_budgets: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25)
    primary_removal_budget: float = 0.15

    fmin: float = 8.0
    fmax: float = 30.0
    target_class_name: str = "left_hand"
    trigger_frequency_hz: float = 23.0
    trigger_amplitude_std: float = 0.55
    trigger_channels: Tuple[str, ...] = ("C3", "Cz", "C4")
    csp_components: int = 8

    crcs_min_class_samples: int = 8
    crcs_min_reference_per_class: int = 3
    crcs_margin_threshold: float = 0.0
    crcs_spectral_weight: float = 0.25
    crcs_spectral_z_threshold: float = 2.5
    crcs_subbands: Tuple[Tuple[float, float], ...] = (
        (8.0, 12.0),
        (12.0, 16.0),
        (16.0, 20.0),
        (20.0, 24.0),
        (24.0, 30.0),
    )
    output_dir: str = "bci_security_phase3_results"


def validate_config(cfg: Config) -> None:
    if not cfg.subjects:
        raise ValueError("At least one subject is required.")
    if not cfg.seeds:
        raise ValueError("At least one seed is required.")
    if not all(0.0 < f < 1.0 for f in cfg.calibration_fractions):
        raise ValueError("Phase-III calibration fractions must satisfy 0 < f < 1.")
    if not all(0.0 < p < 0.5 for p in cfg.poison_rates):
        raise ValueError("Poison rates must lie in (0, 0.5).")
    if not all(0.0 < b < 0.5 for b in cfg.removal_budgets):
        raise ValueError("Removal budgets must lie in (0, 0.5).")
    if cfg.primary_removal_budget not in cfg.removal_budgets:
        raise ValueError("primary_removal_budget must be present in removal_budgets.")
    if cfg.trigger_frequency_hz <= cfg.fmin or cfg.trigger_frequency_hz >= cfg.fmax:
        raise ValueError("Trigger frequency must lie inside the analysis passband.")


PACKAGE_VERSION = "PHASE3-COLAB-3.0"

# -----------------------------------------------------------------------------
# Linear algebra: Euclidean Alignment (EA)
# -----------------------------------------------------------------------------


def _trial_covariance(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Trace-normalized covariance of one trial, x shape = channels x samples."""
    if x.ndim != 2:
        raise ValueError("A trial must have shape (channels, samples).")
    xc = x - x.mean(axis=1, keepdims=True)
    c = (xc @ xc.T) / max(1, xc.shape[1] - 1)
    tr = float(np.trace(c))
    if tr <= eps:
        c = c + eps * np.eye(c.shape[0])
        tr = float(np.trace(c))
    return c / tr


def fit_euclidean_alignment(X: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Fit W = R_bar^{-1/2}; X shape = trials x channels x samples."""
    if X.ndim != 3 or len(X) < 2:
        raise ValueError("X must be 3-D with at least two trials.")
    covs = np.stack([_trial_covariance(x) for x in X], axis=0)
    r_bar = covs.mean(axis=0)
    r_bar = 0.5 * (r_bar + r_bar.T) + eps * np.eye(r_bar.shape[0])
    vals, vecs = np.linalg.eigh(r_bar)
    vals = np.clip(vals, eps, None)
    inv_sqrt = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
    return 0.5 * (inv_sqrt + inv_sqrt.T)


def apply_alignment(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    if X.ndim != 3 or W.ndim != 2:
        raise ValueError("Expected X=(trials,channels,samples), W=(channels,channels).")
    return np.einsum("ij,njt->nit", W, X, optimize=True)


# -----------------------------------------------------------------------------
# Targeted spectral backdoor attack
# -----------------------------------------------------------------------------


def resolve_channel_indices(ch_names: Sequence[str], preferred: Sequence[str]) -> List[int]:
    mapping = {str(name).strip().lower(): i for i, name in enumerate(ch_names)}
    out = [mapping[name.lower()] for name in preferred if name.lower() in mapping]
    if out:
        return sorted(set(out))
    # Safe fallback: central 3 channels if names are unavailable/unexpected.
    n = len(ch_names)
    mid = n // 2
    return sorted(set([max(0, mid - 1), mid, min(n - 1, mid + 1)]))


def add_spectral_trigger(
    X: np.ndarray,
    sfreq: float,
    channel_indices: Sequence[int],
    frequency_hz: float,
    amplitude_std: float,
) -> np.ndarray:
    """Add a deterministic narrow-band trigger scaled to each trial/channel SD."""
    X2 = np.array(X, dtype=np.float64, copy=True)
    n_times = X2.shape[-1]
    t = np.arange(n_times, dtype=np.float64) / float(sfreq)
    wave = np.sin(2.0 * np.pi * float(frequency_hz) * t)

    for idx in channel_indices:
        if idx < 0 or idx >= X2.shape[1]:
            raise IndexError(f"Channel index {idx} out of range for {X2.shape[1]} channels.")
        scale = np.std(X2[:, idx, :], axis=1, ddof=1)
        scale = np.maximum(scale, np.finfo(np.float64).eps)
        X2[:, idx, :] += amplitude_std * scale[:, None] * wave[None, :]
    return X2


def poison_training_set(
    X: np.ndarray,
    y: np.ndarray,
    target_class: int,
    poison_rate: float,
    rng: np.random.Generator,
    sfreq: float,
    channel_indices: Sequence[int],
    frequency_hz: float,
    amplitude_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dirty-label targeted poisoning: trigger non-target trials and relabel target."""
    eligible = np.flatnonzero(y != target_class)
    if len(eligible) == 0:
        raise ValueError("No non-target trials are available for poisoning.")

    n_poison = max(1, int(round(poison_rate * len(y))))
    n_poison = min(n_poison, len(eligible))
    chosen = rng.choice(eligible, size=n_poison, replace=False)

    Xp = np.array(X, copy=True)
    yp = np.array(y, copy=True)
    Xp[chosen] = add_spectral_trigger(
        Xp[chosen], sfreq, channel_indices, frequency_hz, amplitude_std
    )
    yp[chosen] = target_class

    mask = np.zeros(len(y), dtype=bool)
    mask[chosen] = True
    return Xp, yp, mask


# -----------------------------------------------------------------------------
# Generic spectral-outlier sanitization baseline
# -----------------------------------------------------------------------------


def spectral_concentration_score(
    X: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    channel_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Trigger-agnostic trial score.

    It does NOT use the attack frequency. For each trial it measures the strongest
    narrow spectral concentration anywhere inside the analysis band, then takes
    the maximum across selected channels.
    """
    if X.ndim != 3:
        raise ValueError("X must be 3-D.")
    if channel_indices is None:
        Xi = X
    else:
        Xi = X[:, list(channel_indices), :]

    nperseg = min(512, Xi.shape[-1])
    freqs, psd = welch(
        Xi,
        fs=float(sfreq),
        nperseg=nperseg,
        noverlap=nperseg // 2,
        axis=-1,
        detrend="constant",
        scaling="density",
    )
    band = (freqs >= fmin) & (freqs <= fmax)
    if band.sum() < 3:
        raise ValueError("Too few Welch frequency bins in the analysis band.")

    p = np.maximum(psd[..., band], np.finfo(np.float64).tiny)
    concentration = p.max(axis=-1) / p.sum(axis=-1)
    return concentration.max(axis=1)


def robust_sanitize(
    X: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    z_threshold: float,
    max_remove_fraction: float,
    min_class_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Remove high-score spectral outliers using median/MAD with a removal cap."""
    if not (len(X) == len(y) == len(scores)):
        raise ValueError("X, y and scores must have equal length.")

    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= np.finfo(np.float64).eps:
        robust_z = np.zeros_like(scores, dtype=float)
    else:
        robust_z = (scores - median) / robust_sigma

    candidates = np.flatnonzero(robust_z > z_threshold)
    max_remove = int(math.floor(max_remove_fraction * len(y)))
    if max_remove > 0 and len(candidates) > max_remove:
        order = candidates[np.argsort(scores[candidates])[::-1]]
        candidates = order[:max_remove]

    keep = np.ones(len(y), dtype=bool)
    keep[candidates] = False

    # Guarantee enough examples remain in every class.
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        kept_count = int(np.sum(keep[cls_idx]))
        if kept_count < min_class_samples:
            need = min(min_class_samples - kept_count, int(np.sum(~keep[cls_idx])))
            if need > 0:
                removed_cls = cls_idx[~keep[cls_idx]]
                restore = removed_cls[np.argsort(scores[removed_cls])[:need]]
                keep[restore] = True

    stats = {
        "score_median": median,
        "score_mad": mad,
        "removed_count": int((~keep).sum()),
        "removed_fraction": float((~keep).mean()),
    }
    return X[keep], y[keep], keep, stats



# -----------------------------------------------------------------------------
# Phase-II candidate defense:
# Calibration-Referenced Class-Consistency Screening (CRCS)
# -----------------------------------------------------------------------------


def _relative_bandpower_features(
    X: np.ndarray,
    sfreq: float,
    channel_indices: Sequence[int],
    subbands: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """
    Generic trigger-agnostic descriptor.

    For each selected motor channel, compute relative Welch power in fixed
    sub-bands spanning the full 8-30 Hz MI band. The implementation never uses
    the attack frequency. Shape: trials x (channels * subbands).
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (trials, channels, samples).")
    if not channel_indices:
        raise ValueError("At least one channel is required.")
    Xi = X[:, list(channel_indices), :]
    nperseg = min(512, Xi.shape[-1])
    freqs, psd = welch(
        Xi,
        fs=float(sfreq),
        nperseg=nperseg,
        noverlap=nperseg // 2,
        axis=-1,
        detrend="constant",
        scaling="density",
    )
    total_band = (freqs >= min(b[0] for b in subbands)) & (
        freqs <= max(b[1] for b in subbands)
    )
    total = trapezoid(psd[..., total_band], freqs[total_band], axis=-1)
    total = np.maximum(total, np.finfo(np.float64).tiny)

    feats = []
    for lo, hi in subbands:
        mask = (freqs >= float(lo)) & (freqs < float(hi))
        if int(mask.sum()) < 2:
            raise ValueError(f"Too few frequency bins for CRCS subband {(lo, hi)}.")
        bp = trapezoid(psd[..., mask], freqs[mask], axis=-1)
        rel = np.maximum(bp / total, np.finfo(np.float64).tiny)
        feats.append(np.log(rel))
    # trials x channels x bands -> trials x flat features
    F = np.stack(feats, axis=-1).reshape(len(X), -1)
    return np.asarray(F, dtype=np.float64)


def _robust_location_scale(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    med = np.median(F, axis=0)
    mad = np.median(np.abs(F - med), axis=0)
    scale = 1.4826 * mad
    # Avoid zero-MAD dimensions without introducing data-dependent explosions.
    fallback = np.std(F, axis=0, ddof=1) if len(F) > 1 else np.ones(F.shape[1])
    scale = np.where(scale > 1e-8, scale, np.maximum(fallback, 1e-6))
    return med, scale


def _standardize_with_reference(
    F_reference: np.ndarray, F_other: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    med, scale = _robust_location_scale(F_reference)
    return (F_reference - med) / scale, (F_other - med) / scale


def _class_prototypes(F: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
    protos: Dict[int, np.ndarray] = {}
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        if len(idx):
            protos[int(cls)] = np.median(F[idx], axis=0)
    return protos


def _prototype_distances(
    F: np.ndarray, prototypes: Dict[int, np.ndarray]
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for cls, proto in prototypes.items():
        # Mean absolute robust-standardized distance is stable in small calibration sets.
        out[int(cls)] = np.mean(np.abs(F - proto[None, :]), axis=1)
    return out


def _class_mismatch_margin(
    F: np.ndarray, y_claimed: np.ndarray, prototypes: Dict[int, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Margin > 0 means the trial is closer to another class prototype than to its
    claimed class. This is useful for dirty-label poisoning without knowing the
    attack target class.
    """
    dists = _prototype_distances(F, prototypes)
    classes = sorted(prototypes)
    claimed = np.full(len(F), np.nan, dtype=float)
    best_other = np.full(len(F), np.inf, dtype=float)

    for i, cls in enumerate(y_claimed):
        c = int(cls)
        if c not in dists:
            continue
        claimed[i] = dists[c][i]
        alt = [dists[k][i] for k in classes if k != c]
        if alt:
            best_other[i] = float(np.min(alt))
    margin = claimed - best_other
    return margin, claimed, best_other


def calibration_referenced_consistency_screen(
    X_source: np.ndarray,
    y_source_claimed: np.ndarray,
    X_reference: np.ndarray,
    y_reference: np.ndarray,
    sfreq: float,
    channel_indices: Sequence[int],
    subbands: Sequence[Tuple[float, float]],
    max_remove_fraction: float,
    min_class_samples: int,
    min_reference_per_class: int,
    margin_threshold: float,
    spectral_weight: float,
    spectral_z_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    CRCS candidate defense.

    Reference hierarchy:
    1) labeled clean target-session calibration data when every class has enough
       examples;
    2) otherwise the source set itself, using robust medians (zero-calibration
       fallback).

    The screen is attack-frequency agnostic:
    - class-consistency term: source trial vs. class prototypes;
    - spectral term: generic narrow-band concentration relative to the reference
      distribution for the claimed class.

    Samples are removed only when the combined evidence is suspicious, with a
    global removal cap and per-class minimum sample guarantee.
    """
    if len(X_source) != len(y_source_claimed):
        raise ValueError("Source X/y length mismatch.")
    if len(X_reference) != len(y_reference):
        raise ValueError("Reference X/y length mismatch.")

    classes = np.unique(y_source_claimed)
    ref_ok = len(X_reference) > 0 and all(
        np.sum(y_reference == cls) >= min_reference_per_class for cls in classes
    )
    if ref_ok:
        Xref, yref = X_reference, y_reference
        reference_kind = "target_calibration"
    else:
        Xref, yref = X_source, y_source_claimed
        reference_kind = "source_robust_fallback"

    F_ref = _relative_bandpower_features(
        Xref, sfreq, channel_indices, subbands
    )
    F_src = _relative_bandpower_features(
        X_source, sfreq, channel_indices, subbands
    )
    F_ref_z, F_src_z = _standardize_with_reference(F_ref, F_src)
    protos = _class_prototypes(F_ref_z, yref)

    # If a class is missing from the reference, fall back to source prototypes.
    missing = [int(c) for c in classes if int(c) not in protos]
    if missing:
        F_all_ref_z, F_all_src_z = _standardize_with_reference(F_src, F_src)
        source_protos = _class_prototypes(F_all_ref_z, y_source_claimed)
        for c in missing:
            protos[c] = source_protos[c]
        # Keep original source feature coordinates; missing-class fallback is rare.
        # The cap/min-count safeguards prevent catastrophic removal.

    margin, d_claim, _ = _class_mismatch_margin(
        F_src_z, y_source_claimed, protos
    )

    # Generic spectral concentration (does not know trigger frequency).
    src_conc = spectral_concentration_score(
        X_source,
        sfreq=sfreq,
        fmin=min(b[0] for b in subbands),
        fmax=max(b[1] for b in subbands),
        channel_indices=channel_indices,
    )
    ref_conc = spectral_concentration_score(
        Xref,
        sfreq=sfreq,
        fmin=min(b[0] for b in subbands),
        fmax=max(b[1] for b in subbands),
        channel_indices=channel_indices,
    )

    spectral_z = np.zeros(len(X_source), dtype=float)
    for cls in classes:
        src_i = np.flatnonzero(y_source_claimed == cls)
        ref_i = np.flatnonzero(yref == cls)
        if len(ref_i) < 2:
            ref_i = np.arange(len(yref))
        vals = ref_conc[ref_i]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        sigma = max(1.4826 * mad, float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 1e-8)
        spectral_z[src_i] = (src_conc[src_i] - med) / sigma

    # Positive class mismatch is primary; narrow-band excess is secondary.
    positive_margin = np.maximum(margin - float(margin_threshold), 0.0)
    spectral_excess = np.maximum(
        spectral_z - float(spectral_z_threshold), 0.0
    )
    suspicion = positive_margin + float(spectral_weight) * spectral_excess

    candidates = np.flatnonzero(suspicion > 0.0)
    max_remove = int(math.floor(max_remove_fraction * len(y_source_claimed)))
    if max_remove > 0 and len(candidates) > max_remove:
        order = candidates[np.argsort(suspicion[candidates])[::-1]]
        candidates = order[:max_remove]

    keep = np.ones(len(y_source_claimed), dtype=bool)
    keep[candidates] = False

    # Preserve enough examples per class.
    for cls in classes:
        cls_idx = np.flatnonzero(y_source_claimed == cls)
        if int(np.sum(keep[cls_idx])) < min_class_samples:
            need = min_class_samples - int(np.sum(keep[cls_idx]))
            removed_cls = cls_idx[~keep[cls_idx]]
            if len(removed_cls):
                restore = removed_cls[np.argsort(suspicion[removed_cls])[:need]]
                keep[restore] = True

    stats = {
        "reference_target_calibration": float(reference_kind == "target_calibration"),
        "removed_count": int(np.sum(~keep)),
        "removed_fraction": float(np.mean(~keep)),
        "mean_suspicion": float(np.nanmean(suspicion)),
        "max_suspicion": float(np.nanmax(suspicion)),
        "mean_margin": float(np.nanmean(margin)),
        "mean_spectral_z": float(np.nanmean(spectral_z)),
    }
    return X_source[keep], y_source_claimed[keep], keep, stats


def choose_target_alignment(
    alignment_mode: str,
    X_source_reference: np.ndarray,
    X_target_all: np.ndarray,
    X_target_calibration: np.ndarray,
) -> np.ndarray:
    """
    transductive: use all unlabeled target trials (Phase-I setting).
    inductive: use calibration trials only; at 0% calibration use source alignment.
    """
    if alignment_mode == "transductive":
        return fit_euclidean_alignment(X_target_all)
    if alignment_mode == "inductive":
        if len(X_target_calibration) >= 2:
            return fit_euclidean_alignment(X_target_calibration)
        return fit_euclidean_alignment(X_source_reference)
    raise ValueError(f"Unknown alignment_mode={alignment_mode!r}")



# -----------------------------------------------------------------------------
# Classifier and metrics
# -----------------------------------------------------------------------------


class CSPLDA:
    """Small deterministic CSP + shrinkage-LDA baseline for 4-class MI."""

    def __init__(self, n_components: int = 8):
        self.n_components = int(n_components)
        self.csp: Optional[CSP] = None
        self.lda: Optional[LinearDiscriminantAnalysis] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPLDA":
        n_components = max(2, min(self.n_components, X.shape[1]))
        self.csp = CSP(
            n_components=n_components,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False,
            transform_into="average_power",
            rank=None,
        )
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            F = self.csp.fit_transform(X, y)
        self.lda.fit(F, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.csp is None or self.lda is None:
            raise RuntimeError("Model has not been fitted.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            F = self.csp.transform(X)
        return self.lda.predict(F)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def attack_success_rate(
    model: CSPLDA,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    target_class: int,
    target_alignment: np.ndarray,
    sfreq: float,
    channel_indices: Sequence[int],
    frequency_hz: float,
    amplitude_std: float,
) -> float:
    non_target = y_test != target_class
    if non_target.sum() == 0:
        return float("nan")
    X_attack = add_spectral_trigger(
        X_test_raw[non_target],
        sfreq,
        channel_indices,
        frequency_hz,
        amplitude_std,
    )
    X_attack_aligned = apply_alignment(X_attack, target_alignment)
    pred = model.predict(X_attack_aligned)
    return float(np.mean(pred == target_class))


# -----------------------------------------------------------------------------
# Calibration split
# -----------------------------------------------------------------------------


def stratified_calibration_split(
    y: np.ndarray, fraction: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Return calibration/test indices; leaves at least one test trial per class."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must satisfy 0 <= fraction < 1.")

    calib: List[int] = []
    test: List[int] = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        idx = rng.permutation(idx)
        if fraction == 0.0:
            n_cal = 0
        else:
            n_cal = int(round(fraction * len(idx)))
            n_cal = max(1, n_cal)
            n_cal = min(n_cal, len(idx) - 1)
        calib.extend(idx[:n_cal].tolist())
        test.extend(idx[n_cal:].tolist())

    calib_arr = np.asarray(sorted(calib), dtype=int)
    test_arr = np.asarray(sorted(test), dtype=int)
    if len(test_arr) == 0:
        raise RuntimeError("Calibration split left no held-out target data.")
    return calib_arr, test_arr


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------


def load_bnci_subject(
    subject: int, fmin: float, fmax: float
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, float, List[str]]:
    """Load one subject through MOABB. Import is lazy for clearer install errors."""
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "MOABB is required for dataset download/loading. Recommended:\n"
            "  python -m pip install -r requirements_colab.txt\n"
            "Use Python 3.11 or newer."
        ) from exc

    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, fmin=fmin, fmax=fmax)
    epochs, labels, meta = paradigm.get_data(
        dataset=dataset,
        subjects=[int(subject)],
        return_epochs=True,
    )

    # MOABB/MNE returns EEG in volts. Do not rescale: CSP is scale-invariant enough
    # for this benchmark, while trigger amplitude is defined relative to trial SD.
    try:
        X = epochs.get_data(copy=True)
    except TypeError:  # compatibility with older MNE
        X = epochs.get_data().copy()
    y_text = np.asarray(labels).astype(str)
    meta = pd.DataFrame(meta).reset_index(drop=True)
    sfreq = float(epochs.info["sfreq"])
    ch_names = list(epochs.ch_names)

    if not (len(X) == len(y_text) == len(meta)):
        raise RuntimeError("MOABB returned inconsistent data/label/metadata lengths.")
    if "session" not in meta.columns:
        raise RuntimeError("MOABB metadata does not contain a 'session' column.")
    return X, y_text, meta, sfreq, ch_names



# -----------------------------------------------------------------------------
# Phase III helpers: nested poisoning, CRCS ranking, budget application
# -----------------------------------------------------------------------------


def make_nested_poison_order(y: np.ndarray, target_class: int, seed: int) -> np.ndarray:
    """One deterministic non-target permutation; higher poison rates are nested."""
    eligible = np.flatnonzero(y != target_class)
    if len(eligible) == 0:
        raise ValueError("No non-target source trials are available for poisoning.")
    rng = np.random.default_rng(int(seed))
    return rng.permutation(eligible)


def poison_training_set_from_order(
    X: np.ndarray,
    y: np.ndarray,
    target_class: int,
    poison_rate: float,
    poison_order: np.ndarray,
    sfreq: float,
    channel_indices: Sequence[int],
    frequency_hz: float,
    amplitude_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dirty-label attack using the first n entries of a fixed poison order."""
    n_poison = max(1, int(round(float(poison_rate) * len(y))))
    n_poison = min(n_poison, len(poison_order))
    chosen = np.asarray(poison_order[:n_poison], dtype=int)

    Xp = np.array(X, copy=True)
    yp = np.array(y, copy=True)
    Xp[chosen] = add_spectral_trigger(
        Xp[chosen], sfreq, channel_indices, frequency_hz, amplitude_std
    )
    yp[chosen] = target_class
    mask = np.zeros(len(y), dtype=bool)
    mask[chosen] = True
    return Xp, yp, mask


def crcs_suspicion_ranking(
    X_source: np.ndarray,
    y_source_claimed: np.ndarray,
    X_reference: np.ndarray,
    y_reference: np.ndarray,
    sfreq: float,
    channel_indices: Sequence[int],
    subbands: Sequence[Tuple[float, float]],
    min_reference_per_class: int,
    margin_threshold: float,
    spectral_weight: float,
    spectral_z_threshold: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute attack-frequency-agnostic CRCS suspicion once for a poisoned source set.
    The same ranking is then evaluated under several independent removal budgets.
    """
    if len(X_source) != len(y_source_claimed):
        raise ValueError("Source X/y length mismatch.")
    if len(X_reference) != len(y_reference):
        raise ValueError("Reference X/y length mismatch.")

    classes = np.unique(y_source_claimed)
    ref_ok = len(X_reference) > 0 and all(
        np.sum(y_reference == cls) >= min_reference_per_class for cls in classes
    )
    if ref_ok:
        Xref, yref = X_reference, y_reference
        reference_kind = "target_calibration"
    else:
        Xref, yref = X_source, y_source_claimed
        reference_kind = "source_robust_fallback"

    F_ref = _relative_bandpower_features(Xref, sfreq, channel_indices, subbands)
    F_src = _relative_bandpower_features(X_source, sfreq, channel_indices, subbands)
    F_ref_z, F_src_z = _standardize_with_reference(F_ref, F_src)
    protos = _class_prototypes(F_ref_z, yref)

    missing = [int(c) for c in classes if int(c) not in protos]
    if missing:
        F_all_ref_z, _ = _standardize_with_reference(F_src, F_src)
        source_protos = _class_prototypes(F_all_ref_z, y_source_claimed)
        for c in missing:
            protos[c] = source_protos[c]

    margin, _, _ = _class_mismatch_margin(F_src_z, y_source_claimed, protos)

    src_conc = spectral_concentration_score(
        X_source,
        sfreq=sfreq,
        fmin=min(b[0] for b in subbands),
        fmax=max(b[1] for b in subbands),
        channel_indices=channel_indices,
    )
    ref_conc = spectral_concentration_score(
        Xref,
        sfreq=sfreq,
        fmin=min(b[0] for b in subbands),
        fmax=max(b[1] for b in subbands),
        channel_indices=channel_indices,
    )

    spectral_z = np.zeros(len(X_source), dtype=float)
    for cls in classes:
        src_i = np.flatnonzero(y_source_claimed == cls)
        ref_i = np.flatnonzero(yref == cls)
        if len(ref_i) < 2:
            ref_i = np.arange(len(yref))
        vals = ref_conc[ref_i]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        sigma = max(
            1.4826 * mad,
            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            1e-8,
        )
        spectral_z[src_i] = (src_conc[src_i] - med) / sigma

    positive_margin = np.maximum(margin - float(margin_threshold), 0.0)
    spectral_excess = np.maximum(spectral_z - float(spectral_z_threshold), 0.0)
    suspicion = positive_margin + float(spectral_weight) * spectral_excess

    stats = {
        "reference_target_calibration": float(reference_kind == "target_calibration"),
        "candidate_fraction": float(np.mean(suspicion > 0.0)),
        "mean_suspicion": float(np.nanmean(suspicion)),
        "max_suspicion": float(np.nanmax(suspicion)),
        "mean_margin": float(np.nanmean(margin)),
        "mean_spectral_z": float(np.nanmean(spectral_z)),
    }
    return suspicion, stats


def apply_crcs_removal_budget(
    y_source_claimed: np.ndarray,
    suspicion: np.ndarray,
    max_remove_fraction: float,
    min_class_samples: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply one removal cap to a fixed CRCS suspicion ranking."""
    if len(y_source_claimed) != len(suspicion):
        raise ValueError("y/suspicion length mismatch.")
    candidates = np.flatnonzero(suspicion > 0.0)
    max_remove = int(math.floor(float(max_remove_fraction) * len(y_source_claimed)))
    if max_remove > 0 and len(candidates) > max_remove:
        order = candidates[np.argsort(suspicion[candidates])[::-1]]
        candidates = order[:max_remove]

    keep = np.ones(len(y_source_claimed), dtype=bool)
    keep[candidates] = False

    for cls in np.unique(y_source_claimed):
        cls_idx = np.flatnonzero(y_source_claimed == cls)
        n_kept = int(np.sum(keep[cls_idx]))
        if n_kept < min_class_samples:
            need = min_class_samples - n_kept
            removed_cls = cls_idx[~keep[cls_idx]]
            if len(removed_cls):
                restore = removed_cls[np.argsort(suspicion[removed_cls])[:need]]
                keep[restore] = True

    return keep, {
        "removed_count": int(np.sum(~keep)),
        "removed_fraction": float(np.mean(~keep)),
    }


def _fit_evaluate_model(
    X_source_raw: np.ndarray,
    y_source: np.ndarray,
    X_cal_aligned: np.ndarray,
    y_cal: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    target_alignment: np.ndarray,
    cfg: Config,
    sfreq: float,
    motor_channels: Sequence[int],
    target_class: int,
) -> Tuple[Dict[str, float], float]:
    Ws = fit_euclidean_alignment(X_source_raw)
    Xs_a = apply_alignment(X_source_raw, Ws)
    if len(X_cal_aligned):
        X_train = np.concatenate([Xs_a, X_cal_aligned], axis=0)
        y_train = np.concatenate([y_source, y_cal], axis=0)
    else:
        X_train, y_train = Xs_a, y_source

    model = CSPLDA(cfg.csp_components).fit(X_train, y_train)
    pred = model.predict(apply_alignment(X_test_raw, target_alignment))
    metrics = classification_metrics(y_test, pred)
    asr = attack_success_rate(
        model=model,
        X_test_raw=X_test_raw,
        y_test=y_test,
        target_class=target_class,
        target_alignment=target_alignment,
        sfreq=sfreq,
        channel_indices=motor_channels,
        frequency_hz=cfg.trigger_frequency_hz,
        amplitude_std=cfg.trigger_amplitude_std,
    )
    return metrics, asr


# -----------------------------------------------------------------------------
# One pre-specified Phase-III block: subject x seed x calibration
# -----------------------------------------------------------------------------


def evaluate_phase3_block(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    sfreq: float,
    ch_names: Sequence[str],
    subject: int,
    seed: int,
    calibration_fraction: float,
    target_class: int,
    cfg: Config,
) -> List[Dict[str, object]]:
    sessions = sorted(meta["session"].astype(str).unique().tolist())
    if len(sessions) < 2:
        raise RuntimeError(f"Subject {subject}: expected >=2 sessions, got {sessions}.")
    source_session, target_session = sessions[0], sessions[1]

    src_idx = np.flatnonzero(meta["session"].astype(str).to_numpy() == source_session)
    tgt_idx = np.flatnonzero(meta["session"].astype(str).to_numpy() == target_session)
    Xs_raw, ys = X[src_idx], y[src_idx]
    Xt_raw, yt = X[tgt_idx], y[tgt_idx]

    rng_split = np.random.default_rng(seed)
    cal_idx, test_idx = stratified_calibration_split(yt, calibration_fraction, rng_split)
    X_cal_raw, y_cal = Xt_raw[cal_idx], yt[cal_idx]
    X_test_raw, y_test = Xt_raw[test_idx], yt[test_idx]

    motor_channels = resolve_channel_indices(ch_names, cfg.trigger_channels)

    # Strict inductive target alignment: only labeled calibration trials are used.
    Wt = choose_target_alignment(
        alignment_mode="inductive",
        X_source_reference=Xs_raw,
        X_target_all=Xt_raw,
        X_target_calibration=X_cal_raw,
    )
    Xcal_a = apply_alignment(X_cal_raw, Wt)

    common = {
        "subject": int(subject),
        "seed": int(seed),
        "source_session": source_session,
        "target_session": target_session,
        "alignment_mode": "inductive",
        "calibration_fraction": float(calibration_fraction),
        "n_source_original": int(len(Xs_raw)),
        "n_target_calibration": int(len(cal_idx)),
        "n_target_test": int(len(test_idx)),
        "target_class": int(target_class),
        "target_class_name": cfg.target_class_name,
    }

    rows: List[Dict[str, object]] = []

    # Clean reference is fit once per block.
    clean_metrics, clean_asr = _fit_evaluate_model(
        Xs_raw, ys, Xcal_a, y_cal, X_test_raw, y_test, Wt,
        cfg, sfreq, motor_channels, target_class,
    )
    rows.append({
        **common,
        "scenario": "clean",
        "poison_rate": 0.0,
        "removal_budget": np.nan,
        "n_poison": 0,
        "actual_poison_fraction": 0.0,
        "n_source_train": int(len(Xs_raw)),
        "asr": clean_asr,
        **clean_metrics,
    })

    # Fixed order makes 5/10/15/20% poison sets nested for this subject/seed.
    poison_order = make_nested_poison_order(
        ys, target_class, seed=seed + 100_000 + 1000 * int(subject)
    )

    for poison_rate in cfg.poison_rates:
        Xp, yp, poison_mask = poison_training_set_from_order(
            X=Xs_raw,
            y=ys,
            target_class=target_class,
            poison_rate=poison_rate,
            poison_order=poison_order,
            sfreq=sfreq,
            channel_indices=motor_channels,
            frequency_hz=cfg.trigger_frequency_hz,
            amplitude_std=cfg.trigger_amplitude_std,
        )
        n_poison = int(np.sum(poison_mask))
        actual_poison_fraction = float(n_poison / len(ys))

        poison_metrics, poison_asr = _fit_evaluate_model(
            Xp, yp, Xcal_a, y_cal, X_test_raw, y_test, Wt,
            cfg, sfreq, motor_channels, target_class,
        )
        rows.append({
            **common,
            "scenario": "poisoned",
            "poison_rate": float(poison_rate),
            "removal_budget": np.nan,
            "n_poison": n_poison,
            "actual_poison_fraction": actual_poison_fraction,
            "n_source_train": int(len(Xp)),
            "asr": poison_asr,
            **poison_metrics,
        })

        # Compute CRCS ranking once for this poison-rate/calibration block.
        Ws_poison = fit_euclidean_alignment(Xp)
        Xp_desc = apply_alignment(Xp, Ws_poison)
        Xcal_desc = apply_alignment(X_cal_raw, Wt)
        suspicion, rank_stats = crcs_suspicion_ranking(
            X_source=Xp_desc,
            y_source_claimed=yp,
            X_reference=Xcal_desc,
            y_reference=y_cal,
            sfreq=sfreq,
            channel_indices=motor_channels,
            subbands=cfg.crcs_subbands,
            min_reference_per_class=cfg.crcs_min_reference_per_class,
            margin_threshold=cfg.crcs_margin_threshold,
            spectral_weight=cfg.crcs_spectral_weight,
            spectral_z_threshold=cfg.crcs_spectral_z_threshold,
        )

        for budget in cfg.removal_budgets:
            keep, budget_stats = apply_crcs_removal_budget(
                y_source_claimed=yp,
                suspicion=suspicion,
                max_remove_fraction=budget,
                min_class_samples=cfg.crcs_min_class_samples,
            )
            removed = ~keep
            poison_recall = float(np.sum(removed & poison_mask) / max(1, n_poison))
            removal_precision = float(
                np.sum(removed & poison_mask) / max(1, np.sum(removed))
            )
            clean_false_removal_rate = float(
                np.sum(removed & ~poison_mask) / max(1, np.sum(~poison_mask))
            )

            metrics, asr = _fit_evaluate_model(
                Xp[keep], yp[keep], Xcal_a, y_cal, X_test_raw, y_test, Wt,
                cfg, sfreq, motor_channels, target_class,
            )
            rows.append({
                **common,
                "scenario": "crcs",
                "poison_rate": float(poison_rate),
                "removal_budget": float(budget),
                "n_poison": n_poison,
                "actual_poison_fraction": actual_poison_fraction,
                "n_source_train": int(np.sum(keep)),
                "defense_removed_count": int(np.sum(removed)),
                "defense_removed_fraction": float(np.mean(removed)),
                "defense_poison_recall": poison_recall,
                "defense_removal_precision": removal_precision,
                "defense_clean_false_removal_rate": clean_false_removal_rate,
                **{f"defense_{k}": v for k, v in rank_stats.items()},
                **{f"defense_budget_{k}": v for k, v in budget_stats.items()},
                "asr": asr,
                **metrics,
            })

    return rows


# -----------------------------------------------------------------------------
# Statistics and publication-oriented summaries
# -----------------------------------------------------------------------------


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted.tolist()


def matched_rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for b-a; negative favors lower b."""
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    w_pos = float(np.sum(r[d > 0]))
    w_neg = float(np.sum(r[d < 0]))
    denom = w_pos + w_neg
    return float((w_pos - w_neg) / denom) if denom else 0.0


def _paired_subject_test(
    subject_level: pd.DataFrame,
    calibration_fraction: float,
    poison_rate: float,
    scenario_a: str,
    scenario_b: str,
    metric: str,
    primary_budget: float,
) -> Optional[Dict[str, object]]:
    block = subject_level[
        (subject_level["calibration_fraction"] == calibration_fraction)
        & (subject_level["poison_rate"] == poison_rate)
    ]
    A = block[block["scenario"] == scenario_a][["subject", metric]].rename(columns={metric: "a"})
    if scenario_b == "crcs":
        Bsrc = block[
            (block["scenario"] == "crcs")
            & np.isclose(block["removal_budget"].astype(float), primary_budget)
        ]
    else:
        Bsrc = block[block["scenario"] == scenario_b]
    B = Bsrc[["subject", metric]].rename(columns={metric: "b"})
    pair = A.merge(B, on="subject", how="inner").dropna()
    if len(pair) < 5:
        return None
    try:
        stat, p = wilcoxon(pair["a"], pair["b"], zero_method="wilcox", alternative="two-sided")
        stat, p = float(stat), float(p)
    except ValueError:
        stat, p = float("nan"), 1.0
    diff = pair["b"].to_numpy() - pair["a"].to_numpy()
    return {
        "calibration_fraction": float(calibration_fraction),
        "poison_rate": float(poison_rate),
        "scenario_a": scenario_a,
        "scenario_b": scenario_b,
        "metric": metric,
        "removal_budget_b": float(primary_budget) if scenario_b == "crcs" else np.nan,
        "n_subjects": int(len(pair)),
        "mean_a": float(pair["a"].mean()),
        "mean_b": float(pair["b"].mean()),
        "mean_difference_b_minus_a": float(np.mean(diff)),
        "median_difference_b_minus_a": float(np.median(diff)),
        "subjects_improved_b": int(np.sum(diff < 0)) if metric == "asr" else int(np.sum(diff > 0)),
        "wilcoxon_stat": stat,
        "p_raw": p,
        "rank_biserial_b_minus_a": matched_rank_biserial(pair["a"].to_numpy(), pair["b"].to_numpy()),
    }


def make_phase3_statistics(subject_level: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pre-specified families only.

    Primary family (12 tests): poisoned vs CRCS ASR at 15% removal budget across
    3 calibration levels x 4 poison rates. Holm correction is applied only here.

    Secondary family (12 tests): clean vs CRCS accuracy at the same 15% budget.
    """
    primary: List[Dict[str, object]] = []
    secondary: List[Dict[str, object]] = []

    for cal in cfg.calibration_fractions:
        for pr in cfg.poison_rates:
            x = _paired_subject_test(
                subject_level, cal, pr, "poisoned", "crcs", "asr", cfg.primary_removal_budget
            )
            if x is not None:
                primary.append(x)

            # Clean has poison_rate=0 in storage. Build pairing manually with CRCS at pr.
            clean = subject_level[
                (subject_level["calibration_fraction"] == cal)
                & (subject_level["scenario"] == "clean")
            ][["subject", "accuracy"]].rename(columns={"accuracy": "a"})
            crcs = subject_level[
                (subject_level["calibration_fraction"] == cal)
                & (subject_level["poison_rate"] == pr)
                & (subject_level["scenario"] == "crcs")
                & np.isclose(subject_level["removal_budget"].astype(float), cfg.primary_removal_budget)
            ][["subject", "accuracy"]].rename(columns={"accuracy": "b"})
            pair = clean.merge(crcs, on="subject", how="inner").dropna()
            if len(pair) >= 5:
                try:
                    stat, p = wilcoxon(pair["a"], pair["b"], zero_method="wilcox", alternative="two-sided")
                    stat, p = float(stat), float(p)
                except ValueError:
                    stat, p = float("nan"), 1.0
                diff = pair["b"].to_numpy() - pair["a"].to_numpy()
                secondary.append({
                    "calibration_fraction": float(cal),
                    "poison_rate": float(pr),
                    "scenario_a": "clean",
                    "scenario_b": "crcs",
                    "metric": "accuracy",
                    "removal_budget_b": float(cfg.primary_removal_budget),
                    "n_subjects": int(len(pair)),
                    "mean_a": float(pair["a"].mean()),
                    "mean_b": float(pair["b"].mean()),
                    "mean_difference_b_minus_a": float(np.mean(diff)),
                    "median_difference_b_minus_a": float(np.median(diff)),
                    "subjects_improved_b": int(np.sum(diff > 0)),
                    "wilcoxon_stat": stat,
                    "p_raw": p,
                    "rank_biserial_b_minus_a": matched_rank_biserial(pair["a"].to_numpy(), pair["b"].to_numpy()),
                })

    pcols = [
        "calibration_fraction", "poison_rate", "scenario_a", "scenario_b", "metric",
        "removal_budget_b", "n_subjects", "mean_a", "mean_b",
        "mean_difference_b_minus_a", "median_difference_b_minus_a",
        "subjects_improved_b", "wilcoxon_stat", "p_raw",
        "p_holm_family", "rank_biserial_b_minus_a",
    ]
    P = pd.DataFrame(primary)
    S = pd.DataFrame(secondary)
    if len(P):
        P["p_holm_family"] = _holm_adjust(P["p_raw"].to_numpy())
        P = P[pcols]
    else:
        P = pd.DataFrame(columns=pcols)
    if len(S):
        S["p_holm_family"] = _holm_adjust(S["p_raw"].to_numpy())
        S = S[pcols]
    else:
        S = pd.DataFrame(columns=pcols)
    return P, S


def make_phase3_plots(raw: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    crcs = raw[raw["scenario"] == "crcs"].copy()
    clean = raw[raw["scenario"] == "clean"].copy()

    for cal in sorted(crcs["calibration_fraction"].unique()):
        g = crcs[crcs["calibration_fraction"] == cal]
        agg = g.groupby(["poison_rate", "removal_budget"], as_index=False).agg(
            asr=("asr", "mean"), accuracy=("accuracy", "mean")
        )
        clean_acc = float(clean[clean["calibration_fraction"] == cal]["accuracy"].mean())
        agg["accuracy_drop_pp"] = 100.0 * (clean_acc - agg["accuracy"])

        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for pr, q in agg.groupby("poison_rate"):
            q = q.sort_values("removal_budget")
            ax.plot(q["removal_budget"] * 100, q["asr"] * 100, marker="o", label=f"Poison {int(round(pr*100))}%")
        ax.axhline(15, linestyle="--", linewidth=1)
        ax.set_xlabel("CRCS maximum removal budget (%)")
        ax.set_ylabel("Attack success rate (%)")
        ax.set_title(f"Phase III: ASR sensitivity — calibration {int(round(cal*100))}%")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"phase3_asr_budget_calibration_{int(round(cal*100))}.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for pr, q in agg.groupby("poison_rate"):
            q = q.sort_values("removal_budget")
            ax.plot(q["removal_budget"] * 100, q["accuracy_drop_pp"], marker="o", label=f"Poison {int(round(pr*100))}%")
        ax.axhline(2, linestyle="--", linewidth=1)
        ax.set_xlabel("CRCS maximum removal budget (%)")
        ax.set_ylabel("Clean-reference accuracy cost (percentage points)")
        ax.set_title(f"Phase III: utility cost — calibration {int(round(cal*100))}%")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"phase3_accuracy_cost_calibration_{int(round(cal*100))}.png", dpi=220)
        plt.close(fig)

    # Hard-subject 80% calibration plots identified prospectively from Phase II.
    for subject in (3, 6):
        g = crcs[
            (crcs["subject"] == subject)
            & np.isclose(crcs["calibration_fraction"], 0.80)
        ]
        if len(g) == 0:
            continue
        agg = g.groupby(["poison_rate", "removal_budget"], as_index=False)["asr"].mean()
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for pr, q in agg.groupby("poison_rate"):
            q = q.sort_values("removal_budget")
            ax.plot(q["removal_budget"] * 100, q["asr"] * 100, marker="o", label=f"Poison {int(round(pr*100))}%")
        ax.axhline(15, linestyle="--", linewidth=1)
        ax.set_xlabel("CRCS maximum removal budget (%)")
        ax.set_ylabel("Attack success rate (%)")
        ax.set_title(f"Hard-subject analysis: S{subject}, 80% calibration")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"phase3_hard_subject_S{subject}_calibration_80.png", dpi=220)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Full benchmark with resumable checkpoints and ETA
# -----------------------------------------------------------------------------


def _block_mask(df: pd.DataFrame, subject: int, seed: int, cal: float) -> pd.Series:
    return (
        (df["subject"] == int(subject))
        & (df["seed"] == int(seed))
        & np.isclose(df["calibration_fraction"].astype(float), float(cal))
    )


def run_phase3_benchmark(cfg: Config, resume: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    validate_config(cfg)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "phase3_checkpoint.csv"
    checkpoint_config_path = out_dir / "phase3_checkpoint_config.json"

    # Protect resumable runs from silent configuration mismatch.
    cfg_signature = asdict(cfg).copy()
    cfg_signature.pop("output_dir", None)
    cfg_signature_text = json.dumps(cfg_signature, sort_keys=True, indent=2)
    if resume and checkpoint_path.exists() and checkpoint_config_path.exists():
        previous = checkpoint_config_path.read_text(encoding="utf-8")
        if previous != cfg_signature_text:
            raise RuntimeError(
                "Checkpoint configuration does not match the current Phase-III grid. "
                "Use a new output directory or remove the old checkpoint explicitly."
            )
    checkpoint_config_path.write_text(cfg_signature_text, encoding="utf-8")

    rows_per_block = 1 + len(cfg.poison_rates) * (1 + len(cfg.removal_budgets))
    total_blocks = len(cfg.subjects) * len(cfg.seeds) * len(cfg.calibration_fractions)

    if resume and checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        existing = pd.read_csv(checkpoint_path)
        print(f"Resuming from checkpoint with {len(existing)} rows.", flush=True)
    else:
        existing = pd.DataFrame()

    completed = set()
    if len(existing):
        counts = existing.groupby(["subject", "seed", "calibration_fraction"]).size()
        for key, count in counts.items():
            if int(count) == rows_per_block:
                completed.add((int(key[0]), int(key[1]), round(float(key[2]), 6)))

    dataset_manifest: List[Dict[str, object]] = []
    start_time = time.time()
    processed_this_run = 0
    total_done_initial = len(completed)

    for subject in cfg.subjects:
        print(f"\n=== Loading subject {subject} ===", flush=True)
        X, labels_text, meta, sfreq, ch_names = load_bnci_subject(subject, cfg.fmin, cfg.fmax)
        le = LabelEncoder()
        y = le.fit_transform(labels_text)
        if cfg.target_class_name not in le.classes_:
            raise RuntimeError(f"Target class {cfg.target_class_name!r} not found; classes={list(le.classes_)}")
        target_class = int(le.transform([cfg.target_class_name])[0])
        sessions = sorted(meta["session"].astype(str).unique().tolist())
        dataset_manifest.append({
            "subject": int(subject), "n_trials": int(len(X)), "n_channels": int(X.shape[1]),
            "n_times": int(X.shape[2]), "sfreq": float(sfreq),
            "sessions": "|".join(sessions), "classes": "|".join(le.classes_.tolist()),
            "channel_names": "|".join(ch_names),
        })

        for cal in cfg.calibration_fractions:
            for seed in cfg.seeds:
                key = (int(subject), int(seed), round(float(cal), 6))
                block_number = total_done_initial + processed_this_run + 1
                if key in completed:
                    print(
                        f"[skip] Subject {subject} | calibration={cal:.2f} | seed={seed} "
                        f"(checkpoint complete)", flush=True
                    )
                    continue

                elapsed = time.time() - start_time
                if processed_this_run > 0:
                    sec_per_block = elapsed / processed_this_run
                    remaining = total_blocks - (total_done_initial + processed_this_run)
                    eta_min = sec_per_block * remaining / 60.0
                    eta_txt = f" | ETA ~{eta_min:.1f} min"
                else:
                    eta_txt = ""
                print(
                    f"Block {block_number}/{total_blocks} | Subject {subject} | "
                    f"calibration={cal:.2f} | seed={seed}{eta_txt}", flush=True
                )

                block_rows = evaluate_phase3_block(
                    X=X, y=y, meta=meta, sfreq=sfreq, ch_names=ch_names,
                    subject=subject, seed=seed, calibration_fraction=cal,
                    target_class=target_class, cfg=cfg,
                )
                if len(block_rows) != rows_per_block:
                    raise RuntimeError(
                        f"Block produced {len(block_rows)} rows; expected {rows_per_block}."
                    )

                if len(existing):
                    existing = existing.loc[~_block_mask(existing, subject, seed, cal)].copy()
                existing = pd.concat([existing, pd.DataFrame(block_rows)], ignore_index=True)
                existing.to_csv(checkpoint_path, index=False)
                existing.to_csv(out_dir / "trial_level_results.csv", index=False)
                processed_this_run += 1

    raw = existing.copy()
    expected_rows = total_blocks * rows_per_block
    if len(raw) != expected_rows:
        raise RuntimeError(f"Phase III incomplete: {len(raw)} rows, expected {expected_rows}.")

    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "kappa", "asr"]
    summary = (
        raw.groupby(
            ["calibration_fraction", "scenario", "poison_rate", "removal_budget"],
            dropna=False, as_index=False,
        )[metric_cols]
        .agg(["mean", "std", "count"])
    )
    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]
    summary.to_csv(out_dir / "summary_results.csv", index=False)
    pd.DataFrame(dataset_manifest).drop_duplicates("subject").to_csv(
        out_dir / "dataset_manifest.csv", index=False
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    subject_level = (
        raw.groupby(
            ["subject", "calibration_fraction", "scenario", "poison_rate", "removal_budget"],
            dropna=False, as_index=False,
        )[metric_cols]
        .mean()
    )
    subject_level.to_csv(out_dir / "subject_level_results.csv", index=False)

    primary_stats, secondary_stats = make_phase3_statistics(subject_level, cfg)
    primary_stats.to_csv(out_dir / "primary_security_statistics.csv", index=False)
    secondary_stats.to_csv(out_dir / "secondary_utility_statistics.csv", index=False)

    crcs = raw[raw["scenario"] == "crcs"].copy()
    diag_cols = [
        "defense_removed_fraction", "defense_poison_recall",
        "defense_removal_precision", "defense_clean_false_removal_rate",
        "defense_candidate_fraction", "defense_reference_target_calibration",
    ]
    diag = (
        crcs.groupby(["calibration_fraction", "poison_rate", "removal_budget"], as_index=False)[diag_cols]
        .agg(["mean", "std", "count"])
    )
    diag.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        for col in diag.columns.to_flat_index()
    ]
    diag.to_csv(out_dir / "defense_diagnostics.csv", index=False)

    # Clean-reference utility cost and ASR reduction tables.
    clean_ref = raw[raw["scenario"] == "clean"].groupby("calibration_fraction")[["accuracy"]].mean().rename(columns={"accuracy":"clean_accuracy"})
    poisoned_ref = raw[raw["scenario"] == "poisoned"].groupby(["calibration_fraction", "poison_rate"])[["asr", "accuracy"]].mean().rename(columns={"asr":"poisoned_asr", "accuracy":"poisoned_accuracy"})
    crcs_mean = crcs.groupby(["calibration_fraction", "poison_rate", "removal_budget"])[["asr", "accuracy"]].mean().rename(columns={"asr":"crcs_asr", "accuracy":"crcs_accuracy"})
    robustness = crcs_mean.join(poisoned_ref, on=["calibration_fraction", "poison_rate"]).join(clean_ref, on="calibration_fraction").reset_index()
    robustness["asr_absolute_reduction_pp"] = 100.0 * (robustness["poisoned_asr"] - robustness["crcs_asr"])
    robustness["asr_relative_reduction_pct"] = 100.0 * (robustness["poisoned_asr"] - robustness["crcs_asr"]) / np.maximum(robustness["poisoned_asr"], 1e-12)
    robustness["clean_accuracy_cost_pp"] = 100.0 * (robustness["clean_accuracy"] - robustness["crcs_accuracy"])
    robustness.to_csv(out_dir / "robustness_summary.csv", index=False)

    hard = raw[
        raw["subject"].isin([3, 6])
        & (raw["scenario"].isin(["poisoned", "crcs"]))
    ].groupby(
        ["subject", "calibration_fraction", "scenario", "poison_rate", "removal_budget"],
        dropna=False, as_index=False,
    )[metric_cols + ["defense_poison_recall", "defense_removal_precision", "defense_clean_false_removal_rate"]].mean()
    hard.to_csv(out_dir / "hard_subjects_S3_S6.csv", index=False)

    make_phase3_plots(raw, out_dir)
    return raw, summary


# -----------------------------------------------------------------------------
# Deterministic smoke test
# -----------------------------------------------------------------------------


def smoke_test() -> None:
    print("Running Phase-III deterministic synthetic smoke test...")
    rng = np.random.default_rng(123)
    sfreq = 250.0
    n_ch, n_t = 6, 500
    y = np.repeat(np.arange(4), 24)
    X = rng.normal(size=(len(y), n_ch, n_t))
    # Add weak class-specific rhythms to avoid degenerate CSP.
    t = np.arange(n_t) / sfreq
    for cls in range(4):
        X[y == cls, cls % n_ch, :] += 0.35 * np.sin(2*np.pi*(10+3*cls)*t)

    order = make_nested_poison_order(y, target_class=0, seed=777)
    X10, y10, m10 = poison_training_set_from_order(X, y, 0, 0.10, order, sfreq, [0,1,2], 23.0, 0.55)
    X20, y20, m20 = poison_training_set_from_order(X, y, 0, 0.20, order, sfreq, [0,1,2], 23.0, 0.55)
    if not np.all(m10 <= m20):
        raise AssertionError("Nested poisoning smoke test failed.")

    Xref = X[:40]
    yref = y[:40]
    suspicion, stats = crcs_suspicion_ranking(
        X20, y20, Xref, yref, sfreq, [0,1,2],
        ((8,12),(12,16),(16,20),(20,24),(24,30)),
        min_reference_per_class=2, margin_threshold=0.0,
        spectral_weight=0.25, spectral_z_threshold=2.5,
    )
    k10, _ = apply_crcs_removal_budget(y20, suspicion, 0.10, 4)
    k25, _ = apply_crcs_removal_budget(y20, suspicion, 0.25, 4)
    if np.sum(~k25) < np.sum(~k10):
        raise AssertionError("Removal-budget monotonicity smoke test failed.")

    model = CSPLDA(n_components=4).fit(X20[k10], y20[k10])
    pred = model.predict(X[:20])
    if len(pred) != 20:
        raise AssertionError("Classifier smoke test failed.")
    print("Smoke test PASSED.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume completed blocks from checkpoint.")
    p.add_argument("--output-dir", default="bci_security_phase3_results")
    p.add_argument("--subjects", nargs="*", type=int, default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--calibration", nargs="*", type=float, default=None)
    p.add_argument("--poison-rates", nargs="*", type=float, default=None)
    p.add_argument("--removal-budgets", nargs="*", type=float, default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        smoke_test()
        return 0

    if args.quick:
        cfg = Config(
            subjects=(1,), seeds=(42,), calibration_fractions=(0.50, 0.80),
            poison_rates=(0.10, 0.20), removal_budgets=(0.10, 0.15, 0.25),
            primary_removal_budget=0.15, output_dir=args.output_dir,
        )
    else:
        cfg = Config(
            subjects=tuple(args.subjects) if args.subjects else tuple(range(1,10)),
            seeds=tuple(args.seeds) if args.seeds else (11,22,33,44,55),
            calibration_fractions=tuple(args.calibration) if args.calibration else (0.25,0.50,0.80),
            poison_rates=tuple(args.poison_rates) if args.poison_rates else (0.05,0.10,0.15,0.20),
            removal_budgets=tuple(args.removal_budgets) if args.removal_budgets else (0.05,0.10,0.15,0.20,0.25),
            primary_removal_budget=0.15,
            output_dir=args.output_dir,
        )

    set_global_seed(cfg.seeds[0])
    raw, summary = run_phase3_benchmark(cfg, resume=args.resume)
    print("\nPhase III complete.")
    print(f"Primary rows: {len(raw)}")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
