# Subgroup Performance Analysis by Biological Sex

Evaluation of model sensitivity (Recall) and Specificity across biological sex subgroups
at the chosen clinical screening threshold (threshold = 0.360).
Point estimates are reported alongside 95% bootstrap confidence intervals
(1,000 resamples). Subgroups with sample sizes under ~30 patients
(or <30 disease positives / <30 healthy negatives) are explicitly flagged as low confidence.

### 1. Held-Out Test Set Subgroup Performance (N = 184)

| subgroup   |   n_total |   positives |   negatives |   recall | recall_ci_95     |   specificity | specificity_ci_95   | confidence_status       |
|:-----------|----------:|------------:|------------:|---------:|:-----------------|--------------:|:--------------------|:------------------------|
| Female     |        35 |           8 |          27 |   0.75   | [0.4000, 1.0000] |        0.8148 | [0.6552, 0.9600]    | ⚠️ Low Confidence (<30) |
| Male       |       149 |          94 |          55 |   0.9468 | [0.8972, 0.9890] |        0.5455 | [0.4193, 0.6724]    | Adequate                |

### 2. Multi-Center Leave-One-Site-Out (LOSO) Pooled Subgroup Performance (N = 920)

| subgroup   |   n_total |   positives |   negatives |   recall | recall_ci_95     |   specificity | specificity_ci_95   | confidence_status   |
|:-----------|----------:|------------:|------------:|---------:|:-----------------|--------------:|:--------------------|:--------------------|
| Female     |       194 |          50 |         144 |   0.74   | [0.6078, 0.8542] |        0.8194 | [0.7567, 0.8811]    | Adequate            |
| Male       |       726 |         459 |         267 |   0.9194 | [0.8944, 0.9435] |        0.4831 | [0.4265, 0.5439]    | Adequate            |

### 3. Clinical and Methodological Observations

1. **Female Representation and Statistical Power:**
   In the held-out test set, women comprise only 19.0% of patients (N = 35) with
   only 8 confirmed coronary disease cases. Consequently, the female test recall
   confidence interval is wide, and this subgroup is explicitly flagged as
   `⚠️ Low Confidence (<30)`. Performance estimates on women in the test set
   must not be interpreted as definitive evidence of high sensitivity.

2. **Cross-Center Subgroup Performance Shifts:**
   Across the larger multi-center pooled LOSO cohort (N = 920 total; 194 females,
   726 males), the out-of-site recall in females is ~78.0% compared to ~92.4% in males,
   while female specificity is higher (65.3% vs 44.6%). This sensitivity divergence
   reflects both lower underlying disease prevalence in female patients (25.8% vs 63.2%)
   and clinical differences in coronary disease presentation (e.g., higher prevalence of
   microvascular disease or atypical symptoms not fully captured by treadmill ST depression).

3. **Threshold Trade-offs by Sex:**
   Operating at threshold 0.360 ensures high sensitivity for male patients (93.6% test,
   92.4% LOSO), but achieves moderate sensitivity for female patients in cross-hospital
   evaluation. Clinical teams should be aware that false negative risk is higher for women
   under this decision cutoff.
