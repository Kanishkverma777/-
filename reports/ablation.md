# Feature Ablation Study Report

Evaluation of feature subsets under 5-fold x 10-repeat Repeated Stratified CV
(50 splits on training set) and Leave-One-Site-Out (LOSO) cross-validation.
Feature subset selection is evaluated strictly on the **training split**;
the all-920-row LOSO results are reported as descriptive multi-center context.
Imputations are performed strictly inside the pipeline; no missing-indicator features are added.

### Ablation Configurations
- **(A) All 13 Features:** Baseline full feature set (`age, trestbps, chol, thalach, oldpeak, ca, sex, cp, fbs, restecg, exang, slope, thal`).
- **(B) Drop ca & thal (11 Features):** Omits fluoroscopy vessel count (`ca`, 66.4% missing overall, 98.4% missing outside Cleveland) and nuclear scintigraphy thalassemia (`thal`, 52.8% missing overall).
- **(C) Drop ca, thal & slope (10 Features):** Omits `ca`, `thal`, and exercise ST-segment slope (`slope`, 33.6% missing overall).
- **(D) Drop ca, thal, slope, chol & fbs (8 Features):** Further omits serum cholesterol (`chol`, 22.0% missing overall, 100% missing in Switzerland) and fasting blood sugar (`fbs`, 9.8% missing overall, 61.0% missing in Switzerland).

### Quantitative Results Table

| feature_set                                       |   n_features | cv_roc_auc      | cv_pr_auc       | cv_brier        | cv_recall       | train_loso_roc_auc   | train_loso_recall   | full_loso_roc_auc   | full_loso_recall   |
|:--------------------------------------------------|-------------:|:----------------|:----------------|:----------------|:----------------|:---------------------|:--------------------|:--------------------|:-------------------|
| (A) All 13 Features                               |           13 | 0.8764 ± 0.0257 | 0.8841 ± 0.0285 | 0.1398 ± 0.0159 | 0.8461 ± 0.0524 | 0.8153 ± 0.0795      | 0.9088 ± 0.0714     | 0.8037 ± 0.0807     | 0.8899 ± 0.0606    |
| (B) Drop ca & thal (11 Features)                  |           11 | 0.8686 ± 0.0271 | 0.8742 ± 0.0326 | 0.1441 ± 0.0161 | 0.8388 ± 0.0504 | 0.8192 ± 0.0682      | 0.9159 ± 0.0599     | 0.8113 ± 0.0755     | 0.9032 ± 0.0540    |
| (C) Drop ca, thal & slope (10 Features)           |           10 | 0.8699 ± 0.0265 | 0.8746 ± 0.0333 | 0.1440 ± 0.0156 | 0.8344 ± 0.0513 | 0.8115 ± 0.0704      | 0.9136 ± 0.0465     | 0.8081 ± 0.0782     | 0.8936 ± 0.0322    |
| (D) Drop ca, thal, slope, chol & fbs (8 Features) |            8 | 0.8685 ± 0.0255 | 0.8764 ± 0.0316 | 0.1453 ± 0.0152 | 0.8356 ± 0.0468 | 0.8119 ± 0.0710      | 0.9172 ± 0.0342     | 0.8057 ± 0.0810     | 0.9017 ± 0.0198    |

### Clinical Analysis and Selection Rule

1. **CV AUC Differences are Within Fold Variance:**
   The cross-validated ROC-AUC across candidate feature sets spans from
   0.8685 to 0.8764, a difference of only 0.0079,
   which is well within one fold standard deviation
   (±0.0257).
   Omitting features with heavy or non-random missingness does not cause a
   statistically measurable reduction in cross-validated discrimination;
   candidate sets are indistinguishable within CV fold variance.

2. **The Clinical Danger of Imputing `ca` and `thal`:**
   Fluoroscopy major vessels (`ca`) and thallium scintigraphy (`thal`)
   are specialized, invasive or nuclear procedures with extreme non-random
   missingness (e.g. `ca` is 98.4% missing outside Cleveland). Imputing `ca`
   with the training median (0) and `thal` with the modal category (3 = normal)
   silently encodes untested patients as having '0 stenosed vessels' and
   'normal perfusion'—that is, clinically healthy. Rather than 'safely filling'
   missing values, this introduces a systematic optimistic bias, falsely
   assuming that patients who never received advanced diagnostic imaging are
   free of significant coronary disease.

3. **Exercise Stress Test Requirement Across Sets:**
   Sets retain exercise stress markers (`thalach`, `oldpeak`, `exang`).
   Dropping `slope`, `chol`, and `fbs` alongside `ca` and `thal` eliminates
   substances and tests with extreme institutional missingness or zero-fill
   artifacts (e.g. 100% missing cholesterol in Switzerland) while retaining
   the primary quantitative exercise hemodynamics and symptom markers.

4. **Selection Rule and Final Recommendation:**
   - **Rule:** Choose the smallest feature set within 1.0 CV std of
     the best that is not worse on LOSO (evaluated on training split with
     tolerance 0.01).
   - **CV 1-std Cutoff:** Top CV ROC-AUC is 0.8764 ±
     0.0257 ((A) All 13 Features), yielding a 1-std
     tolerance cutoff of 0.8507.
     (D) Drop ca, thal, slope, chol & fbs (8 Features) achieves CV ROC-AUC of 0.8685 ± 0.0255,
     well above the cutoff.
   - **Training LOSO Comparison:** (D) Drop ca, thal, slope, chol & fbs (8 Features) achieves Training LOSO
     ROC-AUC of 0.8119 ± 0.0710,
     which is not worse than (A) All 13 Features (0.8153 ±
     0.0795, difference within 0.01 tolerance).
   - **Conclusion:** **(D) Drop ca, thal, slope, chol & fbs (8 Features) (8 Features)**
     is selected as the production model feature set. It achieves equivalent
     cross-validation and cross-site generalization while eliminating features
     with heavy missingness and recording artifacts (including the 100% missing
     Swiss cholesterol and nuclear scintigraphy), avoiding optimistic bias,
     and relying purely on routine bedside hemodynamics and symptom assessments.
