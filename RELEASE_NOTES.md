# v1.0.0 — Publication reproducibility package

This release is the publication-facing reproducibility package supporting:

**Toward Deployable Brain–Computer Interfaces: A Computational Review and Cross-Session Study of Calibration, Generalization, and Security**

Target journal: *Archives of Computational Methods in Engineering*.

## Contents

- Phase-I, Phase-II, and Phase-III executed notebooks
- extracted executable phase scripts
- pinned scientific Python environment files
- archived phase-level numerical results
- subject- and trial-level outputs
- 14 publication-facing result PNGs rendered from the released numerical tables
- a deterministic publication-figure renderer
- dataset access instructions (raw BCI Competition IV Dataset 2a is not redistributed)
- final revision-stage reproducibility audit and statistical reconciliation
- citation, licensing, and Zenodo metadata

## Reproduction status

The complete revision-stage audit checked 6,075/6,075 Phase-I--III scenario rows:

- Phase I: 540/540 exact
- Phase II: 2,160/2,160 checked; 2,159 exact; 1 discrepant CRCS row / 5 numerical fields
- Phase III: 3,375/3,375 checked; 3,367 exact; 8 discrepant CRCS sensitivity rows / 29 numerical fields
- Combined: 6,066/6,075 rows fully exact; 9 discrepant rows / 34 numerical fields

All eight Phase-III discrepancies occur at removal budgets 20% or 25%. There are zero fresh-versus-archived discrepancies at the pre-specified 15% primary removal budget. Recomputed subject-level Wilcoxon/Holm analyses preserve all primary and secondary inferential decisions.

The result is therefore reported as a **qualified reproduction**, not a byte-identical reproduction.

## Provenance

Exact post-audit scientific supporting snapshot cited by the final manuscript:

`d99befd83ff256e9879e3eb87f6654deb205a1b4`

The executed notebooks, numerical Phase-I--III CSV/JSON outputs, revision-audit ledgers, and phase master scripts in this public package are unchanged from their corresponding audited artifacts. The 14 PNG files in the result folders are publication-facing derivatives regenerated from the released numerical tables using manuscript-aligned typography, palette, axis wording, RGB encoding, and 600-dpi output. The numerical CSV/JSON files remain the canonical result record.

## Verification

Run:

```bash
python scripts/verify_final_revision.py --root .
```

Expected terminal result:

```text
Checks failed: 0
RESULT: FINAL REVISION AUDIT CONSISTENCY PASS
```

Publication-facing figures can be regenerated without rerunning the EEG experiments:

```bash
python -m pip install -r requirements-figures.txt
python scripts/render_publication_figures.py
```

## Dataset

BCI Competition IV Dataset 2a / MOABB `BNCI2014_001` is not redistributed. See `dataset/README.md`.

## Archival DOI

The GitHub release is prepared for archival through Zenodo. The DOI should be added to the manuscript and citation metadata only after Zenodo has minted it.
