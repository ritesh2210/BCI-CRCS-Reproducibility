# Dataset access

This repository does **not** redistribute BCI Competition IV Dataset 2a.

The study uses the benchmark through MOABB as:

```python
from moabb.datasets import BNCI2014_001
BNCI2014_001().get_data(subjects=[1])
```

Study boundary:

- dataset: BCI Competition IV Dataset 2a / BNCI2014_001
- subjects: 9
- sessions: 2 sessions on different days
- classes: 4 motor-imagery classes
- EEG channels used by the benchmark: 22
- sampling rate: 250 Hz
- Session 1 (`0train`): source domain
- Session 2 (`1test`): target domain

The `dataset_manifest.csv` files inside the result folders record the dataset identifier, channel selection, trial counts, and sampling rate used by the released analyses.

Users must obtain the benchmark under its original distribution terms. The author-derived result files in this repository do not include the raw EEG recordings.
