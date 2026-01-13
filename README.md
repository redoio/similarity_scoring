# similarity_scoring
Compute **named sentencing metrics**, score **suitability**, and compare individuals with multiple similarity measures (cosine, Euclidean, Tanimoto/Jaccard, etc.).  
All metrics are **name‑based** and **skip‑if‑missing** (no fabricated defaults). Similarities are always computed on the **intersection of present feature names**.

## What this repo contains
- `config.py` — Paths, column map, defaults, offense lists, **name‑based** metric weights.
- `compute_metrics.py` — Reads CSV/XLSX, normalizes units, classifies offenses, computes **named features** (skip‑if‑missing).
- `sentencing_math.py` — Pure math (no I/O): time decomposition, proportions, frequency/trend, rehab, suitability (name‑based).
- `vector_similarity.py` — Named-vector API (thin re-export): 
  `cosine_similarity_named`, `euclidean_distance_named`, `euclidean_similarity_named`,
  `tanimoto_similarity_named`, `jaccard_similarity_named`.
- `similarity_metrics.py` — Matrix metrics (cosine, euclidean, manhattan, jaccard_binary,
  dice_binary, hamming_binary) **and** named-vector similarities:
  `cosine_similarity_named`, `euclidean_distance_named`, `euclidean_similarity_named`,
  `tanimoto_similarity_named`, `jaccard_similarity_named`.
  Implements the global `MIN_OVERLAP` rule, the “all-zero → similarity=1” rule,
  and skip-if-missing behavior for named metrics.
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
- `DEFAULTS`: configuration knobs for age normalization and frequency bounds.
- `SEVERITY_DECAY_RATE`: decay λ used in the `severity_trend` formula.
- `DEFAULTS["years_elapsed_for_trend"]`: optional override for severity trend horizon (years).
  Precedence:
  1) If commitment date columns exist and dates are present → computed years = first prior → last current.
  2) If `DEFAULTS["years_elapsed_for_trend"]` is not `None` → use that value instead.
  This affects only the `severity_trend` metric.
- `MIN_OVERLAP_FOR_SIMILARITY`: minimum number of overlapping, non-missing features
   required to compute cosine / Euclidean / Tanimoto / jaccard_similarity_named (default = 3).
- `METRIC_WEIGHTS`: **dict by name**; only present features contribute to the score.
   Note: The weight for age is now 1.0 (non-zero) so age contributes to suitability.
   This follows the latest preprint where age is an active metric after normalization.
- `METRIC_DIRECTIONS`: +1 / -1 per metric (e.g. `desc_nonvio_*` = +1, `severity_trend` = -1).

> **Important:** Frequency metrics require **both** a valid exposure window and bounds in config.
> Set `COLS['age_years']` to enable the `age` metric.

## Quick start (CLI)
Print features & suitability for a single ID, and optionally compare two IDs with
cosine / Euclidean / Tanimoto / Jaccard-on-keys:
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
s = \frac{1}{1 + d}
$$

- **Tanimoto (continuous “Jaccard” for real-valued vectors):**  

$$
T = \frac{\sum_i w_i x_i y_i}
         {\sum_i w_i x_i^2 + \sum_i w_i y_i^2 - \sum_i w_i x_i y_i}
$$

- **Jaccard (binary sets with threshold \(\tau\)):**  

$$
J = \frac{|A \cap B|}{|A \cup B|}, \quad
A = \{i \mid x_i > \tau\}, \quad
B = \{i \mid y_i > \tau\}
$$


> All similarities operate on the **intersection of feature names** only (handled internally in `similarity_metrics.py`).  
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
| `severity_trend` | 0.4773 | 0.0000 | 1.0000 |

## Results
- Cosine similarity: **NA**
- Euclidean similarity (unit-interval): **0.4709** (distance d = 1.1236)
- Tanimoto (continuous): **0.0000**
- Jaccard (binary, τ=0.0000): **NA**
- Suitability A: **0.8058** (numerator=1.6116, denominator/out_of=2.0000)
- Suitability B: **0.0000** (numerator=0.0000, denominator/out_of=2.0000)
  - If score = NA and denominator = NA → this ID had **no evaluable metrics**.

> To generate your own example with different IDs, rerun the script as shown in **Quick start**.  
> If you need richer shared feature sets, enable additional metrics (e.g., `age`, `freq_*`) in `config.py` so both IDs have them present.

## Programmatic use
```python
import config as CFG
import compute_metrics as cm
import sentencing_math as sm

from vector_similarity import (
    cosine_similarity_named,
    euclidean_distance_named,
    euclidean_similarity_named,
    tanimoto_similarity_named,
    jaccard_similarity_named,
)

# Show current MIN_OVERLAP for quick debugging
MIN_OVERLAP = getattr(CFG, "MIN_OVERLAP_FOR_SIMILARITY", 3)
print(f"MIN_OVERLAP_FOR_SIMILARITY = {MIN_OVERLAP}")

# Load source tables
demo  = cm.read_table(CFG.PATHS["demographics"])
curr  = cm.read_table(CFG.PATHS["current_commitments"])
prior = cm.read_table(CFG.PATHS["prior_commitments"])

# Take the first two IDs as a simple smoke pair
ids = demo[CFG.COLS["id"]].astype(str).dropna().unique().tolist()[:2]

feats_a, aux_a = cm.compute_features(ids[0], demo, curr, prior, CFG.OFFENSE_LISTS)
feats_b, aux_b = cm.compute_features(ids[1], demo, curr, prior, CFG.OFFENSE_LISTS)

weights    = getattr(CFG, "METRIC_WEIGHTS", {})
directions = getattr(CFG, "METRIC_DIRECTIONS", {})

# Suitability – returns None/NaN if no evaluable metrics
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

# Similarity over intersection of present features (Aparna rules)
cos_ab   = cosine_similarity_named(feats_a, feats_b, weights=weights)
euc_dist = euclidean_distance_named(feats_a, feats_b, weights=weights)
euc_sim  = euclidean_similarity_named(feats_a, feats_b, weights=weights)
tan_ab   = tanimoto_similarity_named(feats_a, feats_b, weights=weights)
jac_ab   = jaccard_similarity_named(feats_a, feats_b, thresh=0.0)

print("\n=== Smoke similarity test between two real IDs ===")
print("ID A:", ids[0])
print("ID B:", ids[1])

print("\n-- Similarities / distances --")
print("cosine_similarity:", cos_ab)
print("euclidean_distance:", euc_dist)
print("euclidean_similarity (1 / (1 + d)):", euc_sim)
print("tanimoto_similarity:", tan_ab)
print("jaccard_similarity_named (active keys):", jac_ab)

print("\n-- Suitability --")
print("A suitability:", score_a, "  (num, den):", num_a, den_a)
print("B suitability:", score_b, "  (num, den):", num_b, den_b)

print("\nDone.")

```

## Notes & tips
- **Skip-if-missing:** metrics are only added when inputs are valid (no silent zero-fills).
- **MIN_OVERLAP_FOR_SIMILARITY:** named-vector similarities
  (`cosine_similarity_named`, `euclidean_distance_named`,
  `tanimoto_similarity_named`, `jaccard_similarity_named`) return `NaN` when there
  are fewer than this many overlapping finite features.
- **All-zero rule:** if there are at least `MIN_OVERLAP_FOR_SIMILARITY`
  overlapping usable keys and all intersecting values are zero,
  similarities return `1.0`. (`euclidean_distance_named` returns `0.0`.)
- For DEV with XLSX files, ensure `openpyxl` is installed.
- To get richer shared feature sets, enable `age` and `freq_*` by providing
  the needed inputs/bounds in `config.py`.


## Edge cases (important)

### Why cosine can be NaN while Euclidean similarity is defined

Consider two named vectors:

```python
test_a = {
  "desc_nonvio_curr": 0.0,
  "desc_nonvio_past": 0.6667,
  "severity_trend": 0.1487,
}

test_b = {
  "desc_nonvio_curr": 0.0,
  "desc_nonvio_past": 0.0,
  "severity_trend": 0.0,
}
```

Overlapping keys = 3 → passes `MIN_OVERLAP_FOR_SIMILARITY = 3`.

**Cosine similarity**

Cosine divides by the product of the vector norms.

`test_b` is an all-zero vector on the overlapping keys → its norm is 0.

The cosine denominator collapses → cosine similarity is undefined.

**Result:** `cosine_similarity_named = NaN`.

**Euclidean distance / similarity**

Euclidean distance between a non-zero vector and a zero vector is well-defined.

Distance `d > 0`, so similarity `1 / (1 + d)` is also well-defined.

**Result:** `euclidean_similarity_named ∈ (0, 1)`.

This difference is intentional and mathematically correct, and explains why cosine may be `NaN` while Euclidean similarity is defined for the same inputs.


### Distinct similarity cases (named vectors)

After filtering to overlapping feature names and dropping invalid values:

| Case | Behavior |
|-----|---------|
| Overlap < `MIN_OVERLAP_FOR_SIMILARITY` | All similarities return **NaN** |
| All overlapping usable values are 0 on both sides | Similarities return **1.0** (distance = 0.0) |
| One side all-zero, other not | Cosine / Tanimoto → **NaN** (one norm/denominator collapses) |
|  | Euclidean distance → defined |
|  | Euclidean similarity `1 / (1 + d)` → defined |
| Jaccard-on-keys | Depends on “active keys” above threshold `τ` (often NaN if none) |
| Normal non-zero overlap | All similarities computed normally |

These rules are enforced consistently across  
`cosine_similarity_named`, `euclidean_*`, `tanimoto_*`, and `jaccard_similarity_named`.


## License
MIT
