# Sentencing Metrics – Config-Driven Implementation

This repository contains a **Python implementation** of the sentencing metrics pipeline,
refactored to follow a **config-driven design** based on Aparna Komarla’s instructions.


## Purpose

The project reproduces and extends the **mathematical framework for evaluating incarceration and sentencing outcomes**:

- **LaTeX to Python)**:Converted LaTeX equations/matrices into clean Python functions (`sentencing_math.py`).
- **Data pipeline**: Built a **config-driven data pipeline** that ingests sentencing datasets 
  (demographics, current commitments, prior commitments) and produces:
  - feature vectors,
  - suitability scores,
  - offense category counts (violent, nonviolent, other, clash).


## Final Workflow

The final system is simplified into **two main scripts + one math module**:

- **`config.py`**
  - Single source of truth.
  - Stores all file paths, column names, defaults, suitability weights, and offense classification lists.

- **`compute_metrics_v2.py`**
  - Main runner.
  - Reads inputs and configuration from `config.py`.
  - Computes features and scores for a given ID.
  - Skips features gracefully if inputs are missing.
  - Prints a structured summary of results.

- **`sentencing_math.py`**
  - Contains the mathematical feature scoring functions.


## File Overview

- **config.py** – Paths, columns, defaults, weights, and offense lists.  
- **compute_metrics_v2.py** – Main execution script (single-ID run).  
- **sentencing_math.py** – Core metrics implementation (time, convictions, trends).  
- **demographics.csv/xlsx** – Demographics dataset.  
- **current_commitments.xlsx** – Current commitments dataset.  
- **prior_commitments.xlsx** – Prior commitments dataset.  

(Other debug/utility scripts from earlier versions are no longer required.)


## Usage

1. Edit `config.py` to point to the correct file paths and update offense lists.

2. Run the metrics script for a given **CDCR ID**:

```bash
python compute_metrics_v2.py --cdcr-id 2cf2a233c4
```

### Example Output

```
=== Sentencing Metrics (config-driven) ===
CDCR ID: 2cf2a233c4
Features: {'age': 0.306, 'desc_nonvio_curr': 1.0, 'desc_nonvio_past': 1.0,
           'freq_violent': 1.0, 'freq_total': 1.0, 'severity_trend': 0.5}
Linear score: 2.306
Counts (current): {'violent': 0, 'nonviolent': 1, 'other': 0, 'clash': 0}
Counts (prior):   {'violent': 0, 'nonviolent': 13, 'other': 0, 'clash': 0}
Age used: 40.0
```


## Notes

- All **auto-detection biases** have been removed.
- Offense classification follows explicit rules in `config.py`:
  - If `nonviolent = "rest"`, all unspecified offenses are treated as nonviolent.
  - If both violent and nonviolent are explicit lists, leftovers fall into `"other"`.
  - Codes appearing in multiple categories are flagged as `"clash"`.
- Offense implications (e.g., attempts) are not included in this version, but can be added later.


## License

MIT License (c) 2025 
