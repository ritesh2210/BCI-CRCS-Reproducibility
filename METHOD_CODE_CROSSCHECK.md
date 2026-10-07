# Manuscript–Code Cross-Check

This note records the implementation details checked against the supplied Phase-I, Phase-II, and Phase-III notebooks/result packages while revising the manuscript. It is intended as a reproducibility aid, not as an additional scientific result.

## Dataset and split

The supplied analyses use BCI Competition IV Dataset 2a through the MOABB BNCI2014_001 interface. Session 1 is the source domain and Session 2 the target domain. The strict-inductive target preprocessing used in the primary Phase-III analysis fits target-domain quantities only from the labeled calibration subset; held-out target trials do not contribute to fitted target preprocessing.

## Decoder

The executed pipeline uses Euclidean alignment, eight-component multiclass CSP with covariance regularization, and shrinkage LDA/LSQR. The classical decoder is held fixed while calibration, poisoning prevalence, and screening budget vary.

## CRCS reference logic

The implemented minimum target reference requirement is three calibration trials per class. Phase-II result diagnostics show that the target-session reference is active from the 10% calibration condition onward. The source-statistics fallback applies to the zero-calibration condition. Phase III uses target references at all three tested calibration levels (25%, 50%, and 80%).

## CRCS spectral descriptor

The code computes Welch PSDs using `nperseg = min(512, epoch_length)` and 50% segment overlap, with constant detrending and density scaling. Relative spectral power is summarized in fixed subbands 8–12, 12–16, 16–20, 20–24, and 24–30 Hz over the aligned coordinates associated with C3, Cz, and C4. Features are robustly standardized using a median and `1.4826 × MAD`, with the implementation's fallback handling for degenerate scales.

The narrow-band concentration component is the maximum PSD-bin contribution inside 8–30 Hz relative to total in-band spectral mass, evaluated across the selected aligned coordinates. It is standardized within class and enters the suspicion score as secondary evidence. The known 23-Hz trigger frequency is not supplied to this computation.

## Euclidean-alignment order around screening

The implementation uses two source-alignment roles that are important to state separately:

1. A provisional source EA transform is fitted to the current source pool and used to construct the CRCS ranking.
2. After the selected source trials are removed, source EA is fitted again on the retained raw source trials before final CSP–LDA training.

For strict-inductive target processing, target EA is fitted from target calibration trials only. The Phase-II transductive comparison intentionally relaxes that target-information boundary for the specified comparison.

## Phase-III design and inference

The Phase-III grid uses three calibration levels, four poisoning prevalences, five removal budgets, nine participants, and five random seeds. The 15% removal budget is the pre-specified primary operating point. Seeds are averaged within participant before paired inference. The primary security family compares poisoned versus CRCS ASR in 12 calibration/prevalence cells; the secondary utility family compares clean versus defended accuracy in the same 12 cells. Holm adjustment is applied separately within the two families.

## Interpretation boundaries retained in the revised paper

The supplied experiment evaluates one deterministic dirty-label spectral backdoor, one public MI benchmark, one principal decoder family, and an offline cross-session setting. The revision therefore avoids presenting CRCS as a universal backdoor guarantee, avoids treating Phase III as external confirmation, and explicitly retains the trusted-reference, attack-family, subject-specific, and removal-capacity limitations.
