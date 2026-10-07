# Reproducibility package — cross-session calibration and backdoor robustness in EEG BCI

This repository supports the manuscript **“Toward Deployable Brain–Computer Interfaces: A Computational Review and Cross-Session Study of Calibration, Generalization, and Security”**, submitted to *Archives of Computational Methods in Engineering*.

It contains the code, notebooks, configurations, derived results, subject-level outputs, and revision-stage reproducibility audit for the original Dataset 2a computational study.

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── LICENSE-DATA.txt
├── environment.yml
├── requirements.txt
├── requirements-figures.txt
├── .zenodo.json
│
├── dataset/
│   └── README.md
├── notebooks/
│   ├── BCI_Phase-1.ipynb
│   ├── BCI_Phase-2.ipynb
│   └── BCI_Phase-3.ipynb
├── scripts/
│   ├── phase1_master.py
│   ├── phase2_master.py
│   ├── phase3_master.py
│   ├── render_publication_figures.py
│   └── verify_final_revision.py
├── phase1_results/
├── phase2_results/
├── phase3_results/
├── revision_audit/
└── METHOD_CODE_CROSSCHECK.md
```

## Dataset

The study uses **BCI Competition IV Dataset 2a**, accessed through MOABB as `BNCI2014_001`.

The benchmark is **not redistributed** in this repository. See [`dataset/README.md`](dataset/README.md) for access instructions and the input boundary used by the study.

## Reproduction status

A complete revision-stage rerun checked all **6,075/6,075** Phase-I–III scenario rows under the matched scientific package stack.

- Phase I: **540/540 exact**
- Phase II: **2,160/2,160 checked; 2,159 exact; 1 discrepant CRCS row / 5 numerical fields**
- Phase III: **3,375/3,375 checked; 3,367 exact; 8 discrepant CRCS sensitivity rows / 29 numerical fields**
- Combined: **6,066/6,075 rows fully exact; 9 discrepant rows / 34 numerical fields**

All eight Phase-III discrepancies occur only at removal budgets **20% or 25%**. There are **zero fresh-versus-archived discrepancies at the pre-specified 15% primary removal budget**. Recomputing the Phase-III subject-level Wilcoxon/Holm analysis from the verified primary-budget rows preserved all primary and secondary inferential decisions.

Accordingly, the result is reported as a **qualified reproduction**, not a byte-identical reproduction.

Detailed evidence is in [`revision_audit/`](revision_audit/).

## Version provenance

The scientific artifacts in this public package were exported from the audited scientific snapshot:

`d99befd83ff256e9879e3eb87f6654deb205a1b4`

The executed notebooks, numerical Phase-I–III CSV/JSON outputs, revision-audit ledgers, and phase master scripts carried into this public repository are unchanged from the corresponding audited artifacts. Manuscript, reviewer, editor, journal-submission, and raw benchmark files are intentionally excluded.

The 14 PNG files under `phase1_results/`, `phase2_results/`, and `phase3_results/` are **publication-facing renderings** generated from the released numerical result tables. They use manuscript-aligned typography, palette, axis wording, RGB encoding, and 600-dpi output for broad viewer compatibility; they are presentation derivatives and are not the canonical numerical record. The underlying CSV/JSON files remain the authoritative result artifacts.

The GitHub release `v1.0.0` identifies the public archival snapshot. The provenance SHA above records the underlying audited scientific state.

## Publication-facing figure regeneration

The publication-facing PNGs can be regenerated **without rerunning the EEG experiments**:

```bash
python -m pip install -r requirements-figures.txt
python scripts/render_publication_figures.py
```

The renderer reads the archived result CSVs and rewrites the 14 result PNGs as RGB images at 600 dpi. It does not alter any numerical result table or audit ledger.

## Quick verification of the final revision audit

The lightweight audit checker uses only the Python standard library:

```bash
python scripts/verify_final_revision.py --root .
```

Expected result:

```text
Checks failed: 0
RESULT: FINAL REVISION AUDIT CONSISTENCY PASS
```

This checker validates the released fresh-rerun correction ledgers. It does **not** substitute for rerunning the full Phase-I–III experiment.

## Full experiment rerun

Create the scientific environment:

```bash
conda env create -f environment.yml
conda activate bci-crcs
```

or:

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then execute:

```bash
python scripts/phase1_master.py
python scripts/phase2_master.py
python scripts/phase3_master.py
```

The dataset is obtained separately through MOABB.

## Environment provenance

The retained executed notebooks report Python **3.13.15** with:

- NumPy 2.1.3
- SciPy 1.16.3
- scikit-learn 1.6.1
- pandas 2.2.3
- Matplotlib 3.10.0
- MNE 1.12.1
- MOABB 1.5.0

The independent revision-stage audit was executed under Python **3.13.5** with the same scientific-library versions. This patch-level distinction is documented rather than collapsed into a single claimed runtime.

## Experimental boundary

The original computational study is bounded to:

- BCI Competition IV Dataset 2a / BNCI2014_001;
- nine subjects and two sessions;
- CSP–LDA with Euclidean alignment;
- one controlled 23-Hz dirty-label spectral trigger;
- a clean labeled target-session calibration reference;
- offline cross-session evaluation.

The package does not establish universal protection against BCI backdoors or external validation beyond this benchmark.

## Licenses

- Code: MIT — see [`LICENSE`](LICENSE)
- Author-derived result tables and audit ledgers: CC BY 4.0 — see [`LICENSE-DATA.txt`](LICENSE-DATA.txt)
- BCI Competition IV Dataset 2a is not redistributed and remains subject to its original terms.

## Citation and archival version

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

A versioned archival DOI is intended to be minted from the final GitHub release through Zenodo. Until that DOI is issued, cite the repository version/release together with the article/manuscript.
