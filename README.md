# similarity_scoring

Config-driven utilities to compute named sentencing metrics, score suitability, and compare individuals via cosine similarity.

## What this repo contains
- `config.py` — Paths, column map, defaults, offense lists, **name-based** metric weights.
- `compute_metrics_v2.py` — Reads CSV/XLSX, normalizes units, classifies offenses, computes **named features** (skip-if-missing).
- `sentencing_math.py` — Pure math (no I/O): time vars, proportions, frequency/trend, rehab, suitability (name-based).
- `vector_similarity.py` — Named-vector helpers: `align_keys`, `cosine_from_named`.
- `similarity_metrics.py` — Matrix metrics (cosine, euclidean, manhattan, jaccard, dice, hamming, gower).
- `run_similarity.py` — CLI to print features + suitability; optional cosine sim to a second ID.

## Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip pandas numpy openpyxl
```

## Configure (edit `config.py`)
- Switch data source: `CFG_PROFILE=DEV` for local files, default `PROD` reads GitHub raw URLs.
- Pin remote data: set `DATA_COMMIT=<git-sha>`.
- Map columns in `COLS` (any `None` -> that metric is skipped).
- Provide explicit offense lists in `OFFENSE_LISTS` (unlisted -> `other` by default).
- Set `METRIC_WEIGHTS` (weights are **dict by name**; add/remove metrics freely).

## Quick start (CLI)
```bash
python run_similarity.py --cdcr-id A1234
python run_similarity.py --cdcr-id A1234 --compare-id B5678
```
Example output:
```
=== A1234 ===
desc_nonvio_curr: 1.000
...
Suitability score: 6.420
(age_value=54.0, pct_completed=40.0, time_outside=215.0)

Cosine similarity (A1234 vs B5678): 0.62
```

## Programmatic use
```python
import config as CFG
import compute_metrics_v2 as cm
import sentencing_math as sm
from vector_similarity import cosine_from_named

demo  = cm.read_table(CFG.PATHS["demographics"])
curr  = cm.read_table(CFG.PATHS["current_commitments"])
prior = cm.read_table(CFG.PATHS["prior_commitments"])

feats1, aux1 = cm.compute_features("A1234", demo, curr, prior, CFG.OFFENSE_LISTS)
score = sm.suitability_score_named(feats1, CFG.METRIC_WEIGHTS)

feats2, _ = cm.compute_features("B5678", demo, curr, prior, CFG.OFFENSE_LISTS)
sim = cosine_from_named(feats1, feats2)
```

## Notes & tips
- **Skip-if-missing**: metrics are only added when inputs are valid (no fabricated values).
- Frequency metrics require: valid exposure window **and** `freq_min_rate`/`freq_max_rate` bounds.
- Set `COLS['age_years']` to enable the `age` metric; otherwise it's skipped.
- For DEV with XLSX files, ensure `openpyxl` is installed.

## License
MIT
