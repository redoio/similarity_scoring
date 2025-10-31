# similarity_scoring
Compute **named sentencing metrics**, score **suitability**, and compare individuals with multiple similarity measures (cosine, Euclidean, Tanimoto/Jaccard, etc.).  
All metrics are **name‑based** and **skip‑if‑missing** (no fabricated defaults). Similarities are always computed on the **intersection of present feature names**.

## What this repo contains
- `config.py` — Paths, column map, defaults, offense lists, **name‑based** metric weights.
- `compute_metrics.py` — Reads CSV/XLSX, normalizes units, classifies offenses, computes **named features** (skip‑if‑missing).
- `sentencing_math.py` — Pure math (no I/O): time decomposition, proportions, frequency/trend, rehab, suitability (name‑based).
- `vector_similarity.py` — Named-vector helpers: align_keys, cosine_from_named, cosine_from_named_weighted.
- `similarity_metrics.py` — Matrix metrics: cosine, euclidean, manhattan, jaccard, dice, hamming, gower.
- `run_similarity.py` — CLI: prints features + suitability for an ID; optional cosine vs a second ID.

## Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip pandas numpy openpyxl
```

## Configure
Edit `config.py`:
- `CFG_PROFILE`: set `DEV` for local files; default `PROD` can read GitHub raw URLs.
- `PATHS`: locations for `demographics`, `current_commitments`, `prior_commitments`.
- `COLS`: column mappings (any `None` disables that metric).
- `OFFENSE_LISTS`: explicit code lists for `violent` / `nonviolent` (unlisted → `other` by design).
- `DEFAULTS`: knobs like `freq_min_rate`/`freq_max_rate`, `trend_years_elapsed`, etc.
- `METRIC_WEIGHTS`: **dict by name**; only present features contribute to the score.

> **Important:** Frequency metrics require **both** a valid exposure window and and bounds in config.
> Set `COLS['age_years']` to enable the `age` metric.

## Quick start (CLI)
Print features & suitability for a single ID (and optionally compare to another):
```bash
python run_similarity.py --cdcr-id A1234
python run_similarity.py --cdcr-id A1234 --compare-id B5678
```

Generate a worked example with formulas for two specific IDs (and save to `docs/`):
```powershell
# Windows PowerShell (use backticks for line breaks)
$env:CFG_PROFILE = "DEV"
python .\make_similarity_example.py `
  --id-a "00173d8423" `
  --id-b "0029029e5b" `
  --use-weights `
  --min-shared 2 `
  --out "docs\README_similarity_example.md"
```
```bash
# macOS/Linux
CFG_PROFILE=DEV python ./make_similarity_example.py \
  --id-a "00173d8423" \
  --id-b "0029029e5b" \
  --use-weights \
  --min-shared 2 \
  --out "docs/README_similarity_example.md"
```

Find candidate pairs that share ≥2 features (so the example is more informative):
```bash
python find_pairs_with_shared_features.py --min-shared 2 --limit 2000
```

## Formulas (LaTeX)

- **Cosine (weighted):**  

$$
\cos(\theta) =
\frac{\sum_i w_i x_i y_i}
     {\sqrt{\sum_i w_i x_i^2} \cdot \sqrt{\sum_i w_i y_i^2}}
$$

(With \( w_i = 1 \) if no weights.)

- **Euclidean distance and a unit-interval similarity:**

$$
d = \sqrt{\sum_i w_i (x_i - y_i)^2}, \quad
s = 1 - \frac{d}{\sqrt{\sum_i w_i}}
$$

- **Tanimoto (continuous “Jaccard” for real-valued vectors):**  

$$
T = \frac{\sum_i w_i x_i y_i}
         {\sum_i w_i x_i^2 + \sum_i w_i y_i^2 - \sum_i w_i x_i y_i}
$$

- **Jaccard (binary sets with threshold \(	au\)):**  

$$
J = \frac{|A \cap B|}{|A \cup B|}, \quad
A = \{i \mid x_i > \tau\}, \quad
B = \{i \mid y_i > \tau\}
$$

> All similarities operate on the **intersection of feature names** only (see `vector_similarity.align_keys`).  
> This guards against inflated similarity from fabricated zeros.

## Worked example (inline)

A real‑data example (two IDs, shared features, formulas, and multiple similarities). The markdown below is produced by `make_similarity_example.py` and kept here for review.

**ID A:** `00173d8423`  |  **ID B:** `0029029e5b`
## Shared feature space
- Features in common: **3**
- Weights: METRIC_WEIGHTS

| feature | x (A) | y (B) | weight |
|---|---:|---:|---:|
| `desc_nonvio_curr` | 0.5000 | 0.0000 | 1.0000 |
| `desc_nonvio_past` | 1.0000 | 0.0000 | 1.0000 |
| `severity_trend` | 0.4773 | 0.5000 | 1.0000 |

## Results
- Cosine similarity: **0.3926**
- Euclidean similarity (unit-interval): **0.3544**
- Tanimoto (continuous): **0.1602**
- Jaccard (binary, τ=0.0000): **0.3333**
- Suitability A: **0.6591** (numerator=1.9773, denominator/out_of=3.0)
- Suitability B: **0.1667** (numerator=0.5000, denominator/out_of=3.0)

> To generate your own example with different IDs, rerun the script as shown in **Quick start**.  
> If you need richer shared feature sets, enable additional metrics (e.g., `age`, `freq_*`) in `config.py` so both IDs have them present.

## Programmatic use
```python
import math
import config as CFG
import compute_metrics as cm
import sentencing_math as sm
from vector_similarity import cosine_from_named
from similarity_metrics import (
    euclidean_distance_named,
    jaccard_on_keys,
    tanimoto_from_named,
)

# load source tables
demo  = cm.read_table(CFG.PATHS["demographics"])
curr  = cm.read_table(CFG.PATHS["current_commitments"])
prior = cm.read_table(CFG.PATHS["prior_commitments"])

ids = demo[CFG.COLS["id"]].astype(str).dropna().unique().tolist()[:2]

feats_a, aux_a = cm.compute_features(ids[0], demo, curr, prior, CFG.OFFENSE_LISTS)
feats_b, aux_b = cm.compute_features(ids[1], demo, curr, prior, CFG.OFFENSE_LISTS)

weights    = getattr(CFG, "METRIC_WEIGHTS", {})
directions = getattr(CFG, "METRIC_DIRECTIONS", {})

# suitability – returns None/NaN if no evaluable metrics
score_a, num_a, den_a = sm.suitability_score_named(
    feats_a,
    weights=weights,
    directions=directions,
    return_parts=True,
    none_if_no_metrics=True,
)
score_b, num_b, den_b = sm.suitability_score_named(
    feats_b,
    weights=weights,
    directions=directions,
    return_parts=True,
    none_if_no_metrics=True,
)

# similarity over intersection of present features
cos_ab   = cosine_from_named(feats_a, feats_b)
euc_dist = euclidean_distance_named(feats_a, feats_b, weights=weights)
tan_ab   = tanimoto_from_named(feats_a, feats_b, weights=weights)
jac_ab   = jaccard_on_keys(feats_a, feats_b, thresh=0.0)

print("cosine:", cos_ab)
print("euclidean_distance:", euc_dist)
print("tanimoto:", tan_ab)
print("jaccard_on_keys:", jac_ab)
print("A suitability:", score_a, num_a, den_a)
print("B suitability:", score_b, num_b, den_b)
```

## Notes & tips
- **Skip‑if‑missing:** metrics are only added when inputs are valid (no silent zero‑fills).
- For DEV with XLSX files, ensure `openpyxl` is installed.
- To get richer shared feature sets, enable `age` and `freq_*` by providing the needed inputs/bounds in `config.py`.

## License
MIT
