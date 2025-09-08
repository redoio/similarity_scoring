#!/usr/bin/env python3   # ensures script runs with Python 3 when executed directly (Linux/Mac)
# -*- coding: utf-8 -*-  # sets file encoding to UTF-8 (safe for special characters)
"""
sentencing_math.py

STRICT math→code conversion of the metrics in:
Aparna Komarla (2025), "Exploring Computational Approaches to Sentencing Reform".

Exposes small, documented functions that implement:
  • Min–max normalization
  • Time variables (inside/outside, % completed)
  • Conviction totals (current/past; violent/non-violent)
  • Descriptive scores (proportions)
  • Trend & frequency scores
  • Rehabilitation scores (opportunity-adjusted per month inside)
  • 10-D vector representation (Section 5.2 / Appendix B)
  • Cosine similarity
  • Linear suitability score (Section 5.4)
  • Rules-based eligibility (“non–non–non”, Section 3)

No CSV/Excel reading. No pandas dependency. Designed to be embedded anywhere.

Author: Taufia Hussain
License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import math

# Utilities 

def safe_div(n: float, d: float) -> float:
    """Division that returns 0.0 when denominator is 0/None."""
    if d is None or d == 0:
        return 0.0
    return float(n) / float(d)

def clip01(x: float) -> float:
    """Clamp a value to the closed interval [0, 1]."""
    return max(0.0, min(1.0, float(x)))

def minmax_norm_scalar(x: float,
                       lo: Optional[float] = None,
                       hi: Optional[float] = None) -> float:
    """
    Min–max normalization of a single value to [0,1].
    If lo/hi are None or degenerate (hi == lo), return 1.0 (constant-series behavior).
    """
    if lo is None or hi is None or hi == lo:
        return 1.0
    return clip01((float(x) - float(lo)) / (float(hi) - float(lo)))

# Time variables 

@dataclass
class TimeInputs:
    """Inputs for time-based metrics (months)."""
    current_sentence_months: float   # sn(vio+nonvio)
    completed_months: float          # sc(vio+nonvio)
    past_time_months: float          # sr(vio+nonvio)
    childhood_months: float = 0.0

def compute_time_vars(t: TimeInputs,
                      months_elapsed_total: Optional[float] = None
                      ) -> Tuple[float, float, float]:
    """
    Returns (time_inside_months, pct_current_completed, time_outside_months).
    If months_elapsed_total is None → time_outside = 0.0.
    """
    time_inside = float(t.past_time_months) + float(t.completed_months)
    pct_completed = 100.0 * safe_div(t.completed_months, t.current_sentence_months)
    if months_elapsed_total is None:
        time_outside = 0.0
    else:
        time_outside = max(0.0, float(months_elapsed_total) - time_inside - float(t.childhood_months))
    return time_inside, pct_completed, time_outside

# Conviction counts 

@dataclass
class ConvictionCounts:
    """Conviction counts by period and type."""
    curr_nonviolent: float
    curr_violent: float
    past_nonviolent: float
    past_violent: float

def aggregate_convictions(c: ConvictionCounts
                          ) -> Tuple[float, float, float, float, float]:
    """
    Returns:
      conv_curr_total, conv_past_total, conv_total,
      conv_violent_total, conv_nonviolent_total
    """
    conv_curr_total = c.curr_nonviolent + c.curr_violent
    conv_past_total = c.past_nonviolent + c.past_violent
    conv_total = conv_curr_total + conv_past_total
    conv_violent_total = c.curr_violent + c.past_violent
    conv_nonviolent_total = c.curr_nonviolent + c.past_nonviolent
    return (conv_curr_total, conv_past_total, conv_total,
            conv_violent_total, conv_nonviolent_total)

# Descriptive scores 

def score_desc_nonvio_curr(curr_nonviolent: float,
                           conv_curr_total: float) -> float:
    """Proportion of non-violent CURRENT convictions."""
    return clip01(safe_div(curr_nonviolent, conv_curr_total))

def score_desc_nonvio_past(past_nonviolent: float,
                           conv_past_total: float) -> float:
    """Proportion of non-violent PAST convictions."""
    return clip01(safe_div(past_nonviolent, conv_past_total))

def score_age_norm(age_value: float,
                   age_min: Optional[float] = None,
                   age_max: Optional[float] = None) -> float:
    """Age normalized to [0,1] using optional bounds."""
    return minmax_norm_scalar(age_value, age_min, age_max)

# Trend & frequency scores 

def score_freq_violent(conv_violent_total: float,
                       time_outside_months: float,
                       min_rate: Optional[float] = None,
                       max_rate: Optional[float] = None) -> float:
    """Normalized frequency of violent offenses: (violent_total / time_outside)."""
    raw = safe_div(conv_violent_total, time_outside_months)
    return minmax_norm_scalar(raw, min_rate, max_rate)

def score_freq_total(conv_total: float,
                     time_outside_months: float,
                     min_rate: Optional[float] = None,
                     max_rate: Optional[float] = None) -> float:
    """Normalized frequency of all offenses: (total / time_outside)."""
    raw = safe_div(conv_total, time_outside_months)
    return minmax_norm_scalar(raw, min_rate, max_rate)

def score_severity_trend(curr_violent_prop: float,
                         past_violent_prop: float,
                         years_elapsed: float = 0.0) -> float:
    """
    Trend of convictions (violent → non-violent or reverse).
    ~ (desc_vio_past - desc_vio_curr)/(years_elapsed + 1) mapped to [0,1].
    """
    raw_delta = (past_violent_prop - curr_violent_prop) / (years_elapsed + 1.0)
    return clip01((raw_delta + 1.0) / 2.0)

# Rehabilitation scores 

@dataclass
class RehabInputs:
    """Raw credit/milestone tallies achieved during incarceration."""
    edu_general_credits: float = 0.0
    edu_advanced_credits: float = 0.0
    rehab_general_credits: float = 0.0
    rehab_advanced_credits: float = 0.0

def _per_month_inside(value: float, time_inside_months: float) -> float:
    """Rate per month inside (opportunity-adjusted)."""
    return safe_div(value, time_inside_months)

def score_edu_general(edu_general_credits: float, time_inside_months: float,
                      lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    return minmax_norm_scalar(_per_month_inside(edu_general_credits, time_inside_months), lo, hi)

def score_edu_advanced(edu_advanced_credits: float, time_inside_months: float,
                       lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    return minmax_norm_scalar(_per_month_inside(edu_advanced_credits, time_inside_months), lo, hi)

def score_rehab_general(rehab_general_credits: float, time_inside_months: float,
                        lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    return minmax_norm_scalar(_per_month_inside(rehab_general_credits, time_inside_months), lo, hi)

def score_rehab_advanced(rehab_advanced_credits: float, time_inside_months: float,
                         lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    return minmax_norm_scalar(_per_month_inside(rehab_advanced_credits, time_inside_months), lo, hi)

# 10-D Vector (5.2) 

@dataclass
class VectorInputs:
    """Inputs required to produce the 10-D feature vector."""
    time: TimeInputs
    counts: ConvictionCounts
    age_value: float
    age_min: Optional[float] = None
    age_max: Optional[float] = None
    rehab: RehabInputs = field(default_factory=RehabInputs)
    months_elapsed_total: Optional[float] = None
    freq_min_rate: Optional[float] = None
    freq_max_rate: Optional[float] = None
    years_elapsed_for_trend: float = 0.0

def build_10d_vector(vin: VectorInputs) -> List[float]:
    """
    Returns list of length 10:
      1 desc_nonvio_curr, 2 desc_nonvio_past, 3 age_score,
      4 freq_violent, 5 freq_total, 6 severity_trend,
      7 edu_general, 8 edu_advanced, 9 rehab_general, 10 rehab_advanced
    """
    # Time
    time_inside, _, time_outside = compute_time_vars(vin.time, vin.months_elapsed_total)

    # Convictions
    conv_curr_total, conv_past_total, conv_total, conv_violent_total, _ = aggregate_convictions(vin.counts)

    # Descriptive
    desc_nonvio_curr = score_desc_nonvio_curr(vin.counts.curr_nonviolent, conv_curr_total)
    desc_nonvio_past = score_desc_nonvio_past(vin.counts.past_nonviolent, conv_past_total)

    # Age
    age_score = score_age_norm(vin.age_value, vin.age_min, vin.age_max)

    # Frequency
    freq_v = score_freq_violent(conv_violent_total, time_outside, vin.freq_min_rate, vin.freq_max_rate)
    freq_t = score_freq_total(conv_total, time_outside, vin.freq_min_rate, vin.freq_max_rate)

    # Trend
    desc_vio_curr = clip01(safe_div(vin.counts.curr_violent, conv_curr_total))
    desc_vio_past = clip01(safe_div(vin.counts.past_violent, conv_past_total))
    severity_trend = score_severity_trend(desc_vio_curr, desc_vio_past, years_elapsed=vin.years_elapsed_for_trend)

    # Rehab
    edu_g = score_edu_general(vin.rehab.edu_general_credits, time_inside)
    edu_a = score_edu_advanced(vin.rehab.edu_advanced_credits, time_inside)
    reh_g = score_rehab_general(vin.rehab.rehab_general_credits, time_inside)
    reh_a = score_rehab_advanced(vin.rehab.rehab_advanced_credits, time_inside)

    return [desc_nonvio_curr, desc_nonvio_past, age_score,
            freq_v, freq_t, severity_trend,
            edu_g, edu_a, reh_g, reh_a]

# Cosine similarity 

def cosine_similarity(u: Sequence[float], v: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    num = sum(ux * vx for ux, vx in zip(u, v))
    nu = math.sqrt(sum(ux * ux for ux in u))
    nv = math.sqrt(sum(vx * vx for vx in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return num / (nu * nv)

# Suitability (Section 5.4) 

@dataclass
class Weights10:
    """Weights for the 10 metrics in the same order as build_10d_vector()."""
    w1: float = 1; w2: float = 1; w3: float = 1; w4: float = 0; w5: float = 0
    w6: float = 0; w7: float = 1; w8: float = 1; w9: float = 1; w10: float = 1
    def as_list(self) -> List[float]:
        return [self.w1, self.w2, self.w3, self.w4, self.w5,
                self.w6, self.w7, self.w8, self.w9, self.w10]

def suitability_score(vec10: Sequence[float], weights: Weights10 = Weights10()) -> float:
    """Linear suitability score = dot(vec10, weights)."""
    ws = weights.as_list()
    return sum(a * b for a, b in zip(vec10, ws))

# Rules-based eligibility (Sec.3) 

def rules_based_eligibility(current_sentence_months: float,
                            completed_months: float,
                            has_disqualifying_offense: bool = False) -> bool:
    """
    Baseline eligibility:
      • Sentence length ≥ 240 months (20 years)
      • Time served ≥ 120 months (10 years)
      • No disqualifying offense
    """
    cond_len = current_sentence_months >= 20 * 12
    cond_served = completed_months >= 10 * 12
    return bool(cond_len and cond_served and (not has_disqualifying_offense))

# Demo 

if __name__ == "__main__":
    # Tiny smoke test (you can remove this block if you want).
    time_a = TimeInputs(360.0, 180.0, 60.0, 0.0)
    counts_a = ConvictionCounts(4, 0, 3, 0)
    vin_a = VectorInputs(time=time_a, counts=counts_a, age_value=80.0, age_min=18.0, age_max=90.0)
    vec = build_10d_vector(vin_a)
    print("Example — 10-D Vector:", [round(x, 3) for x in vec])
    print("Suitability:", round(suitability_score(vec), 3))
