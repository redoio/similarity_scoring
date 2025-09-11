# Sentencing Metrics

This repository provides a Python implementation of a sentencing metrics pipeline.
It is designed for researchers and developers working with incarceration and sentencing data.

## Purpose

The project implements a framework for evaluating incarceration and sentencing outcomes.
It computes key indicators such as:

- Individual feature vectors
- Suitability scores
- Offense category counts (violent, nonviolent, other, clash)
- Age- and time-based metrics

## Workflow

The system is organized into three main components:

- **`defaults.py`**  
  Defines file paths, column names, defaults, suitability weights, and offense classification lists.

- **`compute_metrics_v2.py`**  
  Main execution script. Reads inputs and configuration, computes features and scores for a given ID, 
  and outputs results in a structured format.

- **`sentencing_math.py`**  
  Contains the mathematical feature scoring functions (time, convictions, trends, and descriptive measures).

## File Overview

- `defaults.py` – Paths, columns, defaults, weights, and offense lists  
- `compute_metrics_v2.py` – Main execution script (single-ID run)  
- `sentencing_math.py` – Core metrics implementation (time, convictions, trends)  
- `demographics.csv/xlsx` – Demographics dataset  
- `current_commitments.xlsx` – Current commitments dataset  
- `prior_commitments.xlsx` – Prior commitments dataset  

(Other debug or utility scripts from earlier development are not required.)

## Usage

1. Edit `defaults.py` to set the correct file paths and update offense lists.

2. Run the metrics script for a given **CDCR ID**:

```bash
python compute_metrics_v2.py --cdcr-id 2cf2a233c4
```

### Example Output

```
=== Sentencing Metrics ===
CDCR ID: 2cf2a233c4
Features: {'age': 0.306, 'desc_nonvio_curr': 1.0, 'desc_nonvio_past': 1.0,
           'freq_violent': 1.0, 'freq_total': 1.0, 'severity_trend': 0.5}
Linear score: 2.306
Counts (current): {'violent': 0, 'nonviolent': 1, 'other': 0, 'clash': 0}
Counts (prior):   {'violent': 0, 'nonviolent': 13, 'other': 0, 'clash': 0}
Age used: 40.0
```

## Notes

- Offense classification is explicit and rule-based (`defaults.py`):  
  - If `nonviolent = "rest"`, all unspecified offenses are treated as nonviolent.  
  - If both violent and nonviolent are explicit lists, leftovers fall into `"other"`.  
  - Codes appearing in multiple categories are flagged as `"clash"`.  
- Offense implications (e.g., attempts) can be added in future versions.

## License

MIT License (c) 2025
