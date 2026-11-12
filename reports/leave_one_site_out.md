# Leave-One-Site-Out (LOSO) Cross-Validation Report

Evaluation across the 4 clinical centers using `dataset` strictly as a
grouping variable (never as a feature). Switzerland is flagged as
low-confidence due to extreme class imbalance (nearly all positive cases):

| site                             | n_train    | n_test      | positives    | negatives    | prevalence    | accuracy        | precision       | recall          | specificity     | f1              | roc_auc         | pr_auc          | brier_score     |
|:---------------------------------|:-----------|:------------|:-------------|:-------------|:--------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| Cleveland                        | 616        | 304         | 139          | 165          | 0.457         | 0.7566          | 0.6702          | 0.9209          | 0.6182          | 0.7758          | 0.8675          | 0.8567          | 0.1542          |
| Hungary                          | 627        | 293         | 106          | 187          | 0.362         | 0.7543          | 0.6062          | 0.9151          | 0.6631          | 0.7293          | 0.8831          | 0.8334          | 0.1485          |
| Switzerland (Low Confidence)     | 797        | 123         | 115          | 8            | 0.935         | 0.8618          | 0.9712          | 0.8783          | 0.625           | 0.9224          | 0.7261          | 0.9711          | 0.1744          |
| VA Long Beach                    | 720        | 200         | 149          | 51           | 0.745         | 0.745           | 0.7917          | 0.8926          | 0.3137          | 0.8391          | 0.7461          | 0.8883          | 0.1704          |
| Macro-Average (Site Means ± Std) | 690 (mean) | 230 (mean)  | 127.2 (mean) | 102.8 (mean) | 0.625 ± 0.263 | 0.7794 ± 0.0551 | 0.7598 ± 0.1605 | 0.9017 ± 0.0198 | 0.5550 ± 0.1621 | 0.8167 ± 0.0836 | 0.8057 ± 0.0810 | 0.8874 ± 0.0602 | 0.1619 ± 0.0125 |
| Pooled (All 920 Patients)        | —          | 920 (total) | 509 (total)  | 411 (total)  | 0.553         | 0.7674          | 0.7368          | 0.9018          | 0.6010          | 0.8110          | 0.8467          | 0.8658          | 0.1586          |

### Clinical Observations & Cross-Site Generalizability

1. **Site Recall Range:** At the calibrated threshold of `0.360`,
   sensitivity ranges from `87.8%` (Switzerland (Low Confidence)) to
   `92.1%` (Cleveland) across clinical sites
   (macro-average: `0.9017 ± 0.0198`, pooled: `0.9018`).
2. **Switzerland Low-Confidence Warning & PR-AUC Baseline:**
   Switzerland includes 8 negative patients and 115
   positive patients (prevalence 93.5%). With single-digit
   negative samples, specificity (62.5%) and
   ROC-AUC (0.7261) cannot be reliably estimated and carry
   wide uncertainty. Notably, Switzerland's baseline PR-AUC for an
   uninformative chance classifier equals its prevalence
   (0.935); thus, the observed PR-AUC (0.9711) reflects
   extreme class imbalance rather than superior predictive discrimination.
3. **Prevalence Shifts Across Sites:**
   Baseline disease prevalence ranges from Hungary
   (36.2%) to Switzerland (Low Confidence) (93.5%)
   (overall pooled prevalence: 0.553), driving institutional
   variations in positive predictive value (Hungary 60.6%
   vs. Switzerland (Low Confidence) 97.1%) while sensitivity remains
   consistently high across all centers.
4. **Cross-Site Specificity & False Positive Trade-Off:**
   Operating at the high-sensitivity threshold of `0.360`,
   specificity ranges from VA Long Beach (31.4%) to
   Hungary (66.3%) (macro-average: `0.5550 ± 0.1621`,
   pooled: `0.6010`), reflecting the deliberate screening
   trade-off of accepting false positives to protect patient safety against
   missed coronary disease.
