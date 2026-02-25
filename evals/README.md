# Similarity Metric Evaluation

This folder contains empirical comparisons of cosine, euclidean (similarity form),
and tanimoto metrics on facility-normalized feature vectors.

Evaluation setup:
- 100 sampled individuals
- All valid feature vectors with >= MIN_OVERLAP_FOR_SIMILARITY features
- Metrics compared via:
  - Spearman rank correlation
  - Top-K neighbor overlap (K=20)

Summary results:
- Euclidean vs Tanimoto: avg Spearman ≈ 0.97, Top-K overlap ≈ 0.98
- Cosine vs others: avg Spearman ≈ 0.92–0.95, Top-K overlap ≈ 0.84–0.86

See CSV files for full per-query results.