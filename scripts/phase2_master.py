#!/usr/bin/env python3
"""
Cross-session, calibration-efficient, security-robust MI-BCI benchmark.

Research question
-----------------
How do clean decoding performance and targeted backdoor vulnerability change when
cross-session target-label calibration is reduced, and can calibration-referenced
class-consistency screening suppress dirty-label backdoors without sacrificing
held-out target-session utility?

Dataset
-------
BCI Competition IV Dataset 2a (MOABB: BNCI2014_001): 9 subjects, 2 sessions,
22 EEG channels, 250 Hz, 4 motor-imagery classes.

IMPORTANT
---------
This script is a Phase-II experimental benchmark scaffold. The proposed
calibration-referenced class-consistency screening (CRCS) is a candidate method to
be validated, NOT a novelty claim. Every table/plot is computed from executed
experiments; there are no hard-coded performance values.

Recommended environment: Python 3.11+, MOABB 1.5.0, MNE 1.12.1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PACKAGE_VERSION = "PHASE2-COLAB-2.2"

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid
from scipy.stats import wilcoxon
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
    calibration_fractions: Tuple[float, ...] = (0.0, 0.10, 0.25, 0.50, 0.75, 0.80)
    alignment_modes: Tuple[str, ...] = ("inductive", "transductive")
    fmin: float = 8.0
    fmax: float = 30.0
    target_class_name: str = "left_hand"
    poison_rate: float = 0.10
    trigger_frequency_hz: float = 23.0
    trigger_amplitude_std: float = 0.55
    trigger_channels: Tuple[str, ...] = ("C3", "Cz", "C4")
    csp_components: int = 8

    # Phase-I spectral sanitizer retained as an explicit baseline.
    sanitizer_z_threshold: float = 3.5
    sanitizer_max_remove_fraction: float = 0.15
    min_class_samples_after_sanitize: int = 8

    # Phase-II candidate defense: calibration-referenced class-consistency screening.
    crcs_max_remove_fraction: float = 0.15
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

    output_dir: str = "bci_security_phase2_results"


def validate_config(cfg: Config) -> None:
    if not cfg.subjects:
        raise ValueError("At least one subject is required.")
    if not cfg.seeds:
        raise ValueError("At least one seed is required.")
    if not all(0.0 <= f < 1.0 for f in cfg.calibration_fractions):
        raise ValueError("Calibration fractions must satisfy 0 <= f < 1.")
    if not 0.0 < cfg.poison_rate < 0.5:
        raise ValueError("poison_rate should be in (0, 0.5) for this benchmark.")
    if cfg.fmin <= 0 or cfg.fmax <= cfg.fmin:
        raise ValueError("Invalid frequency band.")
    if cfg.trigger_frequency_hz <= cfg.fmin or cfg.trigger_frequency_hz >= cfg.fmax:
        raise ValueError(
            "Trigger frequency must lie inside the analysis passband so that the "
            "attack is not trivially removed by preprocessing."
        )
    allowed_modes = {"inductive", "transductive"}
    if not cfg.alignment_modes or any(m not in allowed_modes for m in cfg.alignment_modes):
        raise ValueError(
            f"alignment_modes must be drawn from {sorted(allowed_modes)}; "
            f"got {cfg.alignment_modes}"
        )
    if not 0.0 < cfg.crcs_max_remove_fraction < 0.5:
        raise ValueError("crcs_max_remove_fraction must lie in (0, 0.5).")


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
# One subject x one calibration split
# -----------------------------------------------------------------------------


def evaluate_subject_seed_fraction(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    sfreq: float,
    ch_names: Sequence[str],
    subject: int,
    seed: int,
    calibration_fraction: float,
    alignment_mode: str,
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
    cal_idx, test_idx = stratified_calibration_split(
        yt, calibration_fraction, rng_split
    )
    X_cal_raw, y_cal = Xt_raw[cal_idx], yt[cal_idx]
    X_test_raw, y_test = Xt_raw[test_idx], yt[test_idx]

    motor_channels = resolve_channel_indices(ch_names, cfg.trigger_channels)

    # Target alignment is explicitly separated into strict inductive vs transductive.
    Wt = choose_target_alignment(
        alignment_mode=alignment_mode,
        X_source_reference=Xs_raw,
        X_target_all=Xt_raw,
        X_target_calibration=X_cal_raw,
    )
    Xcal_a = apply_alignment(X_cal_raw, Wt) if len(X_cal_raw) else X_cal_raw
    Xtest_a = apply_alignment(X_test_raw, Wt)

    rows: List[Dict[str, object]] = []

    def train_and_record(
        scenario: str,
        X_source_train_raw: np.ndarray,
        y_source_train: np.ndarray,
        defense_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        Ws = fit_euclidean_alignment(X_source_train_raw)
        Xs_a = apply_alignment(X_source_train_raw, Ws)
        if len(Xcal_a):
            X_train = np.concatenate([Xs_a, Xcal_a], axis=0)
            y_train = np.concatenate([y_source_train, y_cal], axis=0)
        else:
            X_train, y_train = Xs_a, y_source_train

        model = CSPLDA(cfg.csp_components).fit(X_train, y_train)
        pred = model.predict(Xtest_a)
        metrics = classification_metrics(y_test, pred)
        asr = attack_success_rate(
            model=model,
            X_test_raw=X_test_raw,
            y_test=y_test,
            target_class=target_class,
            target_alignment=Wt,
            sfreq=sfreq,
            channel_indices=motor_channels,
            frequency_hz=cfg.trigger_frequency_hz,
            amplitude_std=cfg.trigger_amplitude_std,
        )

        row: Dict[str, object] = {
            "subject": int(subject),
            "seed": int(seed),
            "source_session": source_session,
            "target_session": target_session,
            "alignment_mode": alignment_mode,
            "calibration_fraction": float(calibration_fraction),
            "scenario": scenario,
            "n_source_train": int(len(X_source_train_raw)),
            "n_target_calibration": int(len(cal_idx)),
            "n_target_test": int(len(test_idx)),
            "target_class": int(target_class),
            "target_class_name": cfg.target_class_name,
            "asr": asr,
            **metrics,
        }
        if defense_stats:
            row.update({f"defense_{k}": v for k, v in defense_stats.items()})
        rows.append(row)

    # 1) Clean upper/reference baseline.
    train_and_record("clean", Xs_raw, ys)

    # 2) Poisoned source training.
    rng_attack = np.random.default_rng(seed + 100_000 + subject)
    Xp, yp, poison_mask = poison_training_set(
        X=Xs_raw,
        y=ys,
        target_class=target_class,
        poison_rate=cfg.poison_rate,
        rng=rng_attack,
        sfreq=sfreq,
        channel_indices=motor_channels,
        frequency_hz=cfg.trigger_frequency_hz,
        amplitude_std=cfg.trigger_amplitude_std,
    )
    train_and_record("poisoned", Xp, yp)

    # 3) Phase-I generic spectral sanitizer baseline.
    spec_scores = spectral_concentration_score(
        Xp,
        sfreq=sfreq,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        channel_indices=motor_channels,
    )
    Xd, yd, keep_mask, san_stats = robust_sanitize(
        Xp,
        yp,
        spec_scores,
        z_threshold=cfg.sanitizer_z_threshold,
        max_remove_fraction=cfg.sanitizer_max_remove_fraction,
        min_class_samples=cfg.min_class_samples_after_sanitize,
    )
    removed = ~keep_mask
    san_stats = dict(san_stats)
    san_stats["poison_recall"] = float(
        np.sum(removed & poison_mask) / max(1, np.sum(poison_mask))
    )
    san_stats["removal_precision"] = float(
        np.sum(removed & poison_mask) / max(1, np.sum(removed))
    )
    san_stats["clean_false_removal_rate"] = float(
        np.sum(removed & ~poison_mask) / max(1, np.sum(~poison_mask))
    )
    train_and_record("spectral_baseline", Xd, yd, san_stats)

    # 4) Phase-II CRCS candidate defense.
    # Use separately aligned source/calibration data for descriptor comparison.
    Ws_poison = fit_euclidean_alignment(Xp)
    Xp_desc = apply_alignment(Xp, Ws_poison)
    if len(X_cal_raw):
        Xcal_desc = apply_alignment(X_cal_raw, Wt)
        ycal_desc = y_cal
    else:
        Xcal_desc = np.empty((0, Xp.shape[1], Xp.shape[2]), dtype=Xp.dtype)
        ycal_desc = np.empty((0,), dtype=yp.dtype)

    Xc, yc, crcs_keep, crcs_stats = calibration_referenced_consistency_screen(
        X_source=Xp_desc,
        y_source_claimed=yp,
        X_reference=Xcal_desc,
        y_reference=ycal_desc,
        sfreq=sfreq,
        channel_indices=motor_channels,
        subbands=cfg.crcs_subbands,
        max_remove_fraction=cfg.crcs_max_remove_fraction,
        min_class_samples=cfg.crcs_min_class_samples,
        min_reference_per_class=cfg.crcs_min_reference_per_class,
        margin_threshold=cfg.crcs_margin_threshold,
        spectral_weight=cfg.crcs_spectral_weight,
        spectral_z_threshold=cfg.crcs_spectral_z_threshold,
    )

    # CRCS returned aligned kept trials; train_and_record expects raw trials and
    # fits its own source alignment. Recover raw kept trials using the same mask.
    Xc_raw = Xp[crcs_keep]
    yc_raw = yp[crcs_keep]
    crcs_removed = ~crcs_keep
    crcs_stats = dict(crcs_stats)
    crcs_stats["poison_recall"] = float(
        np.sum(crcs_removed & poison_mask) / max(1, np.sum(poison_mask))
    )
    crcs_stats["removal_precision"] = float(
        np.sum(crcs_removed & poison_mask) / max(1, np.sum(crcs_removed))
    )
    crcs_stats["clean_false_removal_rate"] = float(
        np.sum(crcs_removed & ~poison_mask) / max(1, np.sum(~poison_mask))
    )
    train_and_record("crcs", Xc_raw, yc_raw, crcs_stats)

    return rows


# -----------------------------------------------------------------------------
# Full benchmark
# -----------------------------------------------------------------------------


def run_benchmark(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    validate_config(cfg)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    dataset_manifest: List[Dict[str, object]] = []

    for subject in cfg.subjects:
        print(f"\n=== Loading subject {subject} ===", flush=True)
        X, labels_text, meta, sfreq, ch_names = load_bnci_subject(
            subject, cfg.fmin, cfg.fmax
        )
        le = LabelEncoder()
        y = le.fit_transform(labels_text)
        if cfg.target_class_name not in le.classes_:
            raise RuntimeError(
                f"Target class '{cfg.target_class_name}' not found. Classes={list(le.classes_)}"
            )
        target_class = int(le.transform([cfg.target_class_name])[0])

        sessions = sorted(meta["session"].astype(str).unique().tolist())
        dataset_manifest.append(
            {
                "subject": subject,
                "n_trials": len(X),
                "n_channels": X.shape[1],
                "n_times": X.shape[2],
                "sfreq": sfreq,
                "sessions": "|".join(sessions),
                "classes": "|".join(le.classes_.tolist()),
                "channel_names": "|".join(ch_names),
            }
        )

        for alignment_mode in cfg.alignment_modes:
            for fraction in cfg.calibration_fractions:
                for seed in cfg.seeds:
                    print(
                        f"Subject {subject} | alignment={alignment_mode} | "
                        f"calibration={fraction:.2f} | seed={seed}",
                        flush=True,
                    )
                    rows = evaluate_subject_seed_fraction(
                        X=X,
                        y=y,
                        meta=meta,
                        sfreq=sfreq,
                        ch_names=ch_names,
                        subject=subject,
                        seed=seed,
                        calibration_fraction=fraction,
                        alignment_mode=alignment_mode,
                        target_class=target_class,
                        cfg=cfg,
                    )
                    all_rows.extend(rows)

    raw = pd.DataFrame(all_rows)
    if raw.empty:
        raise RuntimeError("No results were produced.")

    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "kappa", "asr"]
    summary = (
        raw.groupby(["alignment_mode", "calibration_fraction", "scenario"], as_index=False)[metric_cols]
        .agg(["mean", "std", "count"])
    )
    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]

    raw_path = out_dir / "trial_level_results.csv"
    summary_path = out_dir / "summary_results.csv"
    manifest_path = out_dir / "dataset_manifest.csv"
    config_path = out_dir / "config.json"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(dataset_manifest).to_csv(manifest_path, index=False)
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    subject_level = (
        raw.groupby(
            ["subject", "alignment_mode", "calibration_fraction", "scenario"],
            as_index=False,
        )[metric_cols]
        .mean()
    )
    subject_level.to_csv(out_dir / "subject_level_results.csv", index=False)
    make_statistical_tests(subject_level).to_csv(
        out_dir / "statistical_tests.csv", index=False
    )

    diag_cols = [
        c for c in [
            "defense_removed_fraction",
            "defense_poison_recall",
            "defense_removal_precision",
            "defense_clean_false_removal_rate",
            "defense_reference_target_calibration",
        ] if c in raw.columns
    ]
    if diag_cols:
        diagnostics = (
            raw[raw["scenario"].isin(["spectral_baseline", "crcs"])]
            .groupby(
                ["alignment_mode", "calibration_fraction", "scenario"],
                as_index=False,
            )[diag_cols]
            .mean()
        )
        diagnostics.to_csv(out_dir / "defense_diagnostics.csv", index=False)

    make_plots(raw, out_dir)
    print(f"\nSaved results to: {out_dir.resolve()}")
    return raw, summary


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


def make_statistical_tests(subject_level: pd.DataFrame) -> pd.DataFrame:
    """
    Paired Wilcoxon tests use subjects, not repeated seed-runs, as inferential units.
    This avoids pseudoreplication. Holm correction is applied within the produced
    family of pairwise tests.
    """
    tests: List[Dict[str, object]] = []
    comparisons = [
        ("poisoned", "crcs", "asr"),
        ("spectral_baseline", "crcs", "asr"),
        ("poisoned", "crcs", "accuracy"),
        ("clean", "crcs", "accuracy"),
    ]
    for mode in sorted(subject_level["alignment_mode"].unique()):
        for frac in sorted(subject_level["calibration_fraction"].unique()):
            block = subject_level[
                (subject_level["alignment_mode"] == mode)
                & (subject_level["calibration_fraction"] == frac)
            ]
            for a, b, metric in comparisons:
                A = block[block["scenario"] == a][["subject", metric]].rename(
                    columns={metric: "a"}
                )
                B = block[block["scenario"] == b][["subject", metric]].rename(
                    columns={metric: "b"}
                )
                pair = A.merge(B, on="subject", how="inner").dropna()
                if len(pair) < 5:
                    continue
                diff = pair["b"].to_numpy() - pair["a"].to_numpy()
                try:
                    stat, p = wilcoxon(
                        pair["a"].to_numpy(),
                        pair["b"].to_numpy(),
                        zero_method="wilcox",
                        alternative="two-sided",
                    )
                    stat = float(stat)
                    p = float(p)
                except ValueError:
                    stat, p = float("nan"), 1.0
                tests.append(
                    {
                        "alignment_mode": mode,
                        "calibration_fraction": float(frac),
                        "scenario_a": a,
                        "scenario_b": b,
                        "metric": metric,
                        "n_subjects": int(len(pair)),
                        "mean_a": float(pair["a"].mean()),
                        "mean_b": float(pair["b"].mean()),
                        "mean_difference_b_minus_a": float(np.mean(diff)),
                        "median_difference_b_minus_a": float(np.median(diff)),
                        "wilcoxon_stat": stat,
                        "p_raw": p,
                    }
                )
    stat_columns = [
        "alignment_mode",
        "calibration_fraction",
        "scenario_a",
        "scenario_b",
        "metric",
        "n_subjects",
        "mean_a",
        "mean_b",
        "mean_difference_b_minus_a",
        "median_difference_b_minus_a",
        "wilcoxon_stat",
        "p_raw",
        "p_holm",
    ]
    if not tests:
        # Quick mode may contain only one subject, so inferential tests are
        # intentionally unavailable. Return headers rather than a zero-byte CSV.
        return pd.DataFrame(columns=stat_columns)

    out = pd.DataFrame(tests)
    out["p_holm"] = _holm_adjust(out["p_raw"].to_numpy())
    return out[stat_columns]


def make_plots(raw: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    grouped = (
        raw.groupby(["alignment_mode", "calibration_fraction", "scenario"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), asr=("asr", "mean"))
    )

    for mode in sorted(grouped["alignment_mode"].unique()):
        gm = grouped[grouped["alignment_mode"] == mode]

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for scenario, g in gm.groupby("scenario"):
            g = g.sort_values("calibration_fraction")
            ax.plot(
                g["calibration_fraction"] * 100,
                g["accuracy"] * 100,
                marker="o",
                label=scenario,
            )
        ax.set_xlabel("Target-session labeled calibration (%)")
        ax.set_ylabel("Held-out target-session accuracy (%)")
        ax.set_title(f"Cross-session accuracy vs. calibration — {mode}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"accuracy_vs_calibration_{mode}.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for scenario in ["poisoned", "spectral_baseline", "crcs"]:
            g = gm[gm["scenario"] == scenario].sort_values("calibration_fraction")
            if not g.empty:
                ax.plot(
                    g["calibration_fraction"] * 100,
                    g["asr"] * 100,
                    marker="o",
                    label=scenario,
                )
        ax.set_xlabel("Target-session labeled calibration (%)")
        ax.set_ylabel("Targeted attack success rate, ASR (%)")
        ax.set_title(f"Backdoor vulnerability vs. calibration — {mode}")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"asr_vs_calibration_{mode}.png", dpi=300)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Synthetic smoke test (does not download BCI data)
# -----------------------------------------------------------------------------


def smoke_test() -> None:
    print("Running deterministic synthetic smoke test...")
    set_global_seed(7)
    rng = np.random.default_rng(7)
    n_trials, n_ch, n_t, sfreq = 80, 8, 500, 250.0
    X = rng.normal(size=(n_trials, n_ch, n_t))
    y = np.repeat(np.arange(4), n_trials // 4)

    # Add class-specific weak rhythms so CSP has something non-random to learn.
    t = np.arange(n_t) / sfreq
    for i in range(n_trials):
        cls = int(y[i])
        X[i, cls % n_ch] += 0.35 * np.sin(2 * np.pi * (10 + 2 * cls) * t)

    W = fit_euclidean_alignment(X)
    Xa = apply_alignment(X, W)
    if Xa.shape != X.shape or not np.isfinite(Xa).all():
        raise AssertionError("Alignment smoke test failed.")

    rng2 = np.random.default_rng(9)
    Xp, yp, pm = poison_training_set(
        X, y, target_class=0, poison_rate=0.10, rng=rng2, sfreq=sfreq,
        channel_indices=[1, 2, 3], frequency_hz=23.0, amplitude_std=0.6
    )
    if pm.sum() == 0 or not np.all(yp[pm] == 0):
        raise AssertionError("Poisoning smoke test failed.")

    scores = spectral_concentration_score(Xp, sfreq, 8.0, 30.0, [1, 2, 3])
    Xd, yd, keep, _ = robust_sanitize(Xp, yp, scores, 3.5, 0.15, 5)
    if len(Xd) != len(yd) or len(keep) != len(Xp):
        raise AssertionError("Sanitization smoke test failed.")

    cal, test = stratified_calibration_split(y, 0.25, np.random.default_rng(3))
    if set(cal).intersection(set(test)) or len(test) == 0:
        raise AssertionError("Calibration split smoke test failed.")

    # Synthetic CRCS test with a small clean calibration reference.
    ref_idx = np.concatenate([np.flatnonzero(y == c)[:5] for c in np.unique(y)])
    Xc, yc, keep_c, stats_c = calibration_referenced_consistency_screen(
        X_source=Xp,
        y_source_claimed=yp,
        X_reference=X[ref_idx],
        y_reference=y[ref_idx],
        sfreq=sfreq,
        channel_indices=[1, 2, 3],
        subbands=((8, 12), (12, 16), (16, 20), (20, 24), (24, 30)),
        max_remove_fraction=0.15,
        min_class_samples=5,
        min_reference_per_class=3,
        margin_threshold=0.0,
        spectral_weight=0.25,
        spectral_z_threshold=2.5,
    )
    if len(keep_c) != len(Xp) or len(Xc) != len(yc):
        raise AssertionError("CRCS smoke test failed.")

    model = CSPLDA(6).fit(Xa[:60], y[:60])
    pred = model.predict(Xa[60:])
    if len(pred) != 20:
        raise AssertionError("Classifier smoke test failed.")

    # Alignment-mode smoke check.
    Wi = choose_target_alignment("inductive", X[:40], X[40:], X[40:50])
    Wt = choose_target_alignment("transductive", X[:40], X[40:], X[40:50])
    if Wi.shape != Wt.shape:
        raise AssertionError("Alignment-mode smoke test failed.")

    print("Smoke test PASSED.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke-test", action="store_true", help="Run synthetic checks only.")
    p.add_argument("--quick", action="store_true", help="Run subject 1, seed 42, calib 50/80%, both alignment modes.")
    p.add_argument("--output-dir", default="bci_security_results")
    p.add_argument("--subjects", nargs="*", type=int, default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--calibration", nargs="*", type=float, default=None)
    p.add_argument("--alignment-modes", nargs="*", choices=["inductive", "transductive"], default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        smoke_test()
        return 0

    if args.quick:
        cfg = Config(
            subjects=(1,),
            seeds=(42,),
            calibration_fractions=(0.50, 0.80),
            alignment_modes=("inductive", "transductive"),
            output_dir=args.output_dir,
        )
    else:
        cfg = Config(
            subjects=tuple(args.subjects) if args.subjects else tuple(range(1, 10)),
            seeds=tuple(args.seeds) if args.seeds else (11, 22, 33, 44, 55),
            calibration_fractions=tuple(args.calibration)
            if args.calibration
            else (0.0, 0.10, 0.25, 0.50, 0.75, 0.80),
            alignment_modes=tuple(args.alignment_modes)
            if args.alignment_modes
            else ("inductive", "transductive"),
            output_dir=args.output_dir,
        )

    set_global_seed(cfg.seeds[0])
    raw, summary = run_benchmark(cfg)
    print("\nSummary:\n")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
