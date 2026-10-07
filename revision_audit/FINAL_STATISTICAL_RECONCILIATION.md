# BCI-1 Final Statistical Reconciliation

## Scope
This audit follows completion of the full revision-stage rerun:
- Phase I: 540/540 rows checked; 540 exact.
- Phase II: 2160/2160 rows checked; 2159 exact; 1 CRCS row differs in 5 numerical fields.
- Phase III: 3375/3375 rows checked; 3367 exact; 8 CRCS sensitivity rows differ in 29 numerical fields.
- Combined: 6075/6075 rows checked; 6066 exact; 9 discrepant rows; 34 numerical field differences.

The result is a **qualified reproduction**, not byte-identical reproduction.

## Phase-III primary inferential boundary
All eight Phase-III fresh-versus-archived discrepant rows occur at removal budgets B=0.20 or B=0.25.
There are **zero discrepancies at the pre-specified primary budget B=0.15**.

Therefore the exact seed-averaged subject values supporting the primary and secondary B=0.15 inference are unchanged in the fresh rerun.

## Wilcoxon / Holm / paired-difference verification
The 12 poisoned-versus-CRCS ASR tests were recomputed from the exact subject-level values with two-sided paired Wilcoxon signed-rank tests and Holm adjustment across the 12-cell primary family. The archived p-values and matched rank-biserial effects are reproduced exactly.

In particular, the 50% calibration / 20% poisoning primary cell is:
- W = 7.0
- p_raw = 0.07421875
- p_Holm = 0.27343750
- matched rank-biserial = -0.6888888889
- subjects improved = 7/9

Seven of 12 primary comparisons remain significant after Holm correction; none of the 12 secondary clean-accuracy comparisons survives its separate Holm correction. Table 16 subject-level paired differences remain valid.

## Bootstrap confidence-interval audit
The manuscript/response states that 95% percentile bootstrap intervals for the mean paired subject difference were generated as follows:
- nine paired subject values,
- 100,000 resamples with replacement,
- NumPy `default_rng(20260930)`,
- RNG seeded once,
- 12 cells processed in ascending calibration and poisoning order.

This procedure was re-executed under the pinned NumPy 2.1.3 environment using the exact subject-level differences. Only the first printed interval matches the previous manuscript at two decimals. The remaining 11 intervals do not reproduce exactly.

Correct intervals from the declared deterministic procedure are:

| Cal. | Poison | Correct 95% percentile CI (pp) | Previous manuscript CI (pp) |
|---:|---:|---:|---:|
|25%|5%|[-84.91, -34.68]|[-84.91, -34.68]|
|25%|10%|[-61.03, -2.87]|[-61.21, -2.94]|
|25%|15%|[-55.71, 26.57]|[-55.88, 26.94]|
|25%|20%|[0.91, 37.48]|[0.92, 37.49]|
|50%|5%|[-88.95, -62.63]|[-88.87, -62.57]|
|50%|10%|[-81.54, -47.51]|[-81.52, -47.53]|
|50%|15%|[-78.35, -37.82]|[-78.46, -37.86]|
|50%|20%|[-19.12, -1.56]|[-19.20, -1.63]|
|80%|5%|[-82.43, -48.47]|[-82.43, -48.62]|
|80%|10%|[-75.08, -41.48]|[-75.19, -41.43]|
|80%|15%|[-74.02, -38.04]|[-74.02, -38.10]|
|80%|20%|[-24.60, -4.18]|[-24.66, -4.02]|

This correction does **not change any inferential decision or substantive conclusion**. The bootstrap intervals are descriptive and the Wilcoxon/Holm decision rule is unchanged.

## Color rule
Because these CI and fresh-rerun corrections were identified independently during the reproducibility audit, their numerical corrections are NORMAL BLACK in the revised manuscript, even where the surrounding table or text was originally added in RED in direct response to reviewer requests.

## Required reproducibility disclosure
A complete revision-stage rerun checked all 6,075 Phase-I--III scenario rows under the matched scientific package stack. Phase I reproduced exactly; Phase II contained one discrepant CRCS row, and Phase III contained eight discrepant CRCS sensitivity rows. Overall, 6,066/6,075 rows reproduced completely, with 34 numerical-field differences across nine rows. All eight Phase-III discrepancies occurred at 20% or 25% removal budgets, with no fresh-versus-archived discrepancy at the pre-specified 15% primary budget. Recomputing the Phase-III subject-level Wilcoxon/Holm analysis from the verified primary-budget rows preserved all primary and secondary inferential decisions. The archived outputs are retained for provenance and the fresh rerun is supplied separately; the result is therefore reported as a qualified rather than byte-identical reproduction.
