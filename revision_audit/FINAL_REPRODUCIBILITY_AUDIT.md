# BCI-1 Final Reproducibility Audit for Frozen Snapshot

## Status
**Qualified reproduction.**

A complete revision-stage rerun checked **6,075/6,075** Phase-I--III scenario rows under the matched scientific package stack.

- Phase I: **540/540 exact**
- Phase II: **2,160/2,160 checked; 2,159 exact, 1 discrepant CRCS row, 5 numerical fields**
- Phase III: **3,375/3,375 checked; 3,367 exact, 8 discrepant CRCS sensitivity rows, 29 numerical fields**
- Combined: **6,066/6,075 rows fully exact; 9 discrepant rows; 34 numerical-field differences**

All eight Phase-III discrepancies occur only at **20% or 25% removal budgets**. There are **zero fresh-versus-archived discrepancies at the pre-specified 15% primary removal budget**.

Accordingly, the primary and secondary subject-level Wilcoxon/Holm analyses were recomputed from the verified primary-budget rows and all inferential decisions were preserved.

## Pinned scientific environment
- Python 3.13.5
- NumPy 2.1.3
- SciPy 1.16.3
- scikit-learn 1.6.1
- pandas 2.2.3
- matplotlib 3.10.0
- MNE 1.12.1
- MOABB 1.5.0
- CPU execution

## Phase-II confirmed discrepancy
Subject A03, seed 33, 10% calibration, strict-inductive CRCS:
- archived clean accuracy: 85.00%
- fresh clean accuracy: 85.38%
- archived ASR: 2.56%
- fresh ASR: 2.05%
- five numerical fields differ in the row.

At manuscript aggregate precision this changes:
- Phase-II 10% CRCS ASR: **11.19% -> 11.18%**
- spectral-to-CRCS class-reference gain: **+1.80 pp -> +1.81 pp**
- CRCS clean accuracy remains 66.44% at two decimals.

## Phase-III mismatch boundary
Eight CRCS sensitivity rows differ under fresh execution. None is at B=15%.

Displayed higher-budget ASR corrections include:
- 25/10/B25: 10.99 -> 11.03%
- 25/15/B25: 14.50 -> 14.55%
- 25/20/B20: 69.29 -> 69.31%
- 25/20/B25: 25.49 -> 25.40%
- 50/20/B20: 59.88 -> 59.90%
- 50/20/B25: 18.60 -> 18.56%

These propagate to the sensitivity, Pareto, residual-contamination and multi-objective displays. The minimizing budget B* does not change in any condition.

## Bootstrap reconciliation
The reviewer-requested descriptive bootstrap was rerun exactly as declared:
- nine paired subject values
- 100,000 percentile-bootstrap resamples
- NumPy `default_rng(20260930)`
- seeded once
- 12 cells processed in ascending calibration then poisoning order.

Only the first printed Table-15 CI reproduced unchanged at two decimals. Eleven intervals required small BLACK numerical corrections. No Wilcoxon statistic, raw p-value, Holm-adjusted p-value, matched rank-biserial effect, improved-subject count, or Table-16 paired subject difference changed.

## Provenance rule
Archived outputs are retained unchanged for provenance. Fresh-rerun discrepancies are not silently overwritten; they are documented separately in this audit and the correction ledgers.

## Claim boundary
This audit supports a **qualified**, not byte-identical, reproduction. It does not change the manuscript's bounded experimental claim: Dataset 2a, nine subjects, CSP--LDA, one controlled 23-Hz spectral dirty-label trigger, and a clean labeled target-session calibration reference.
