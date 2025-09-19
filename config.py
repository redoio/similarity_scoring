# config.py
import os
import math

# Paths (profiles)
PROFILE = os.getenv("CFG_PROFILE", "PROD")  # "DEV" or "PROD"
COMMIT_SHA = os.getenv("DATA_COMMIT", "main")  # pin for reproducibility

PATHS_PROD = {
    "demographics":        f"https://raw.githubusercontent.com/redoio/offenses_data/{COMMIT_SHA}/data/demographics.csv",
    "prior_commitments":   f"https://raw.githubusercontent.com/redoio/offenses_data/{COMMIT_SHA}/data/prior_commitments.csv",
    "current_commitments": f"https://raw.githubusercontent.com/redoio/offenses_data/{COMMIT_SHA}/data/current_commitments.xlsx",
    # optional:
    "offense_codes":       f"https://raw.githubusercontent.com/redoio/resentencing_data_initiative/{COMMIT_SHA}/eligibility_model/code/offense_codes.xlsx",
}

PATHS_DEV = {
    "demographics":        "/data/local/demographics.csv",
    "prior_commitments":   "/data/local/prior_commitments.csv",
    "current_commitments": "/data/local/current_commitments.xlsx",
    "offense_codes":       "/data/local/offense_codes.xlsx",
}

PATHS = PATHS_PROD if PROFILE == "PROD" else PATHS_DEV


# Columns (schema map)
COLS = {
    "id": "cdcno",
    "age_years": None,               # set if present; else fallback/NaN
    "dob": None,
    "reference_date": None,
    "current_sentence": "aggregate sentence in months",  # years/days ok; code converts to months
    "completed_time":  "time served in years",           # unit-aware conversion
    "past_time":       None,                             # optional
    "current_offense_text": "offense",
    "prior_offense_text":   "offense",
    "current_category_text": "offense category",         # not used by compute script
    "prior_category_text":   "offense category",
}


# Defaults / behavior 
DEFAULTS = {
    # Missing values
    "missing_numeric": math.nan,

    # Required time fields for computing time features
    "require_time_fields": ("current_sentence", "completed_time"),

    # Exposure modeling / time assumptions
    "childhood_months": 0.0,             # used by sentencing_math.TimeInputs
    "months_elapsed_total": None,        # optional exposure window (months) for frequency; None → treat as 0 in math

    # Age normalization
    "age_min": 18.0,
    "age_max": 90.0,
    "age_fallback_years": math.nan,      # or a number if you prefer

    # Frequency normalization (rates per month outside)
    "freq_min_rate": None,
    "freq_max_rate": None,

    # Trend scaling
    "trend_years_elapsed": 10.0,

    # Optional normalization bounds for per-month rehab/education rates
    "rehab_norm_bounds": {
        "edu_general":   (None, None),
        "edu_advanced":  (None, None),
        "rehab_general": (None, None),
        "rehab_advanced":(None, None),
    },
}


#  Offense classification 
OFFENSE_LISTS = {
    "violent": ["187", "211", "245"],  # examples; replace with your canonical codes
    "nonviolent": "rest",              # or explicit list like ["459", "10851"]
}


#  Eligibility thresholds 
# Used by sentencing_math.rules_based_eligibility_from_cfg(...)
ELIGIBILITY = {
    "min_sentence_months": 240,   # e.g., 20 years
    "min_completed_months": 120,  # e.g., 10 years
}


#  Weights (name-based) 
# Name→weight; matches keys produced by build_metrics_named (order not required).
WEIGHTS_10D = {
    "desc_nonvio_curr": 1.0,
    "desc_nonvio_past": 1.0,
    "age": 1.0,
    "freq_violent": 0.0,
    "freq_total": 0.0,
    "severity_trend": 0.0,
    "edu_general": 1.0,
    "edu_advanced": 1.0,
    "rehab_general": 1.0,
    "rehab_advanced": 1.0,
}

# Preferred alias used by sentencing_math.suitability_score_named(...)
METRIC_WEIGHTS = WEIGHTS_10D
