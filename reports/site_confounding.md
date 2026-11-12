# Clinical Center Confounding and Missingness Analysis

Evaluation of institutional demographic differences, sex distribution shifts,
feature missingness rates, and database recording artifacts across the 4 cohorts
(Cleveland, Hungary, Switzerland, VA Long Beach; N = 920 total records).

### 1. Sex Distribution and Disease Prevalence by Clinical Site

| site              |   total_n |   males |   male_share_pct |   male_positives |   male_disease_rate_pct |   females |   female_share_pct |   female_positives |   female_disease_rate_pct |   chol_zero_count |   chol_zero_share_pct |
|:------------------|----------:|--------:|-----------------:|-----------------:|------------------------:|----------:|-------------------:|-------------------:|--------------------------:|------------------:|----------------------:|
| Cleveland         |       304 |     207 |             68.1 |              114 |                    55.1 |        97 |               31.9 |                 25 |                      25.8 |                 0 |                   0   |
| Hungary           |       293 |     212 |             72.4 |               94 |                    44.3 |        81 |               27.6 |                 12 |                      14.8 |                 0 |                   0   |
| Switzerland       |       123 |     113 |             91.9 |              105 |                    92.9 |        10 |                8.1 |                 10 |                     100   |               123 |                 100   |
| VA Long Beach     |       200 |     194 |             97   |              146 |                    75.3 |         6 |                3   |                  3 |                      50   |                49 |                  24.5 |
| Total (All Sites) |       920 |     726 |             78.9 |              459 |                    63.2 |       194 |               21.1 |                 50 |                      25.8 |               172 |                  18.7 |

### 2. Feature Missingness and Numerical Zero Rates Across Clinical Sites

| feature   | Cleveland_missing   | Cleveland_zero   | Hungary_missing   | Hungary_zero   | Switzerland_missing   | Switzerland_zero   | VA Long Beach_missing   | VA Long Beach_zero   | total_missing   | total_zero   |
|:----------|:--------------------|:-----------------|:------------------|:---------------|:----------------------|:-------------------|:------------------------|:---------------------|:----------------|:-------------|
| age       | 0 (0.0%)            | 0 (0.0%)         | 0 (0.0%)          | 0 (0.0%)       | 0 (0.0%)              | 0 (0.0%)           | 0 (0.0%)                | 0 (0.0%)             | 0 (0.0%)        | 0 (0.0%)     |
| trestbps  | 0 (0.0%)            | 0 (0.0%)         | 1 (0.3%)          | 0 (0.0%)       | 2 (1.6%)              | 0 (0.0%)           | 57 (28.5%)              | 1 (0.5%)             | 60 (6.5%)       | 1 (0.1%)     |
| chol      | 0 (0.0%)            | 0 (0.0%)         | 23 (7.8%)         | 0 (0.0%)       | 123 (100.0%)          | 123 (100.0%)       | 56 (28.0%)              | 49 (24.5%)           | 202 (22.0%)     | 172 (18.7%)  |
| thalach   | 0 (0.0%)            | 0 (0.0%)         | 1 (0.3%)          | 0 (0.0%)       | 1 (0.8%)              | 0 (0.0%)           | 53 (26.5%)              | 0 (0.0%)             | 55 (6.0%)       | 0 (0.0%)     |
| oldpeak   | 0 (0.0%)            | 100 (32.9%)      | 0 (0.0%)          | 188 (64.2%)    | 6 (4.9%)              | 42 (34.1%)         | 56 (28.0%)              | 40 (20.0%)           | 62 (6.7%)       | 370 (40.2%)  |
| ca        | 5 (1.6%)            | 176 (57.9%)      | 290 (99.0%)       | 3 (1.0%)       | 118 (95.9%)           | 0 (0.0%)           | 198 (99.0%)             | 2 (1.0%)             | 611 (66.4%)     | 181 (19.7%)  |
| sex       | 0 (0.0%)            | 0 (0.0%)         | 0 (0.0%)          | 0 (0.0%)       | 0 (0.0%)              | 0 (0.0%)           | 0 (0.0%)                | 0 (0.0%)             | 0 (0.0%)        | 0 (0.0%)     |
| cp        | 0 (0.0%)            | 0 (0.0%)         | 0 (0.0%)          | 0 (0.0%)       | 0 (0.0%)              | 0 (0.0%)           | 0 (0.0%)                | 0 (0.0%)             | 0 (0.0%)        | 0 (0.0%)     |
| fbs       | 0 (0.0%)            | 259 (85.2%)      | 8 (2.7%)          | 265 (90.4%)    | 75 (61.0%)            | 43 (35.0%)         | 7 (3.5%)                | 125 (62.5%)          | 90 (9.8%)       | 692 (75.2%)  |
| restecg   | 0 (0.0%)            | 0 (0.0%)         | 1 (0.3%)          | 0 (0.0%)       | 1 (0.8%)              | 0 (0.0%)           | 0 (0.0%)                | 0 (0.0%)             | 2 (0.2%)        | 0 (0.0%)     |
| exang     | 0 (0.0%)            | 205 (67.4%)      | 1 (0.3%)          | 203 (69.3%)    | 1 (0.8%)              | 68 (55.3%)         | 53 (26.5%)              | 52 (26.0%)           | 55 (6.0%)       | 528 (57.4%)  |
| slope     | 1 (0.3%)            | 0 (0.0%)         | 189 (64.5%)       | 0 (0.0%)       | 17 (13.8%)            | 0 (0.0%)           | 102 (51.0%)             | 0 (0.0%)             | 309 (33.6%)     | 0 (0.0%)     |
| thal      | 3 (1.0%)            | 0 (0.0%)         | 265 (90.4%)       | 0 (0.0%)       | 52 (42.3%)            | 0 (0.0%)           | 166 (83.0%)             | 0 (0.0%)             | 486 (52.8%)     | 0 (0.0%)     |

### 3. Breakdown of the 202 Serum Cholesterol (`chol`) Missing Values

| site              |   total_n |   explicit_nan_count |   explicit_nan_pct |   zero_count |   zero_pct |   total_missing_count |   total_missing_pct | clinical_imputation_impact                                                   |
|:------------------|----------:|---------------------:|-------------------:|-------------:|-----------:|----------------------:|--------------------:|:-----------------------------------------------------------------------------|
| Cleveland         |       304 |                    0 |                0   |            0 |        0   |                     0 |                 0   | Complete laboratory assays (0% missing)                                      |
| Hungary           |       293 |                   23 |                7.8 |            0 |        0   |                    23 |                 7.8 | Explicit NaN recorded (7.8%); imputed with training median                   |
| Switzerland       |       123 |                    0 |                0   |          123 |      100   |                   123 |               100   | 100% database zero-fill artifact; imputed with training median (239.0 mg/dL) |
| VA Long Beach     |       200 |                    7 |                3.5 |           49 |       24.5 |                    56 |                28   | Mixed explicit NaN (3.5%) & zero-fill (24.5%); imputed with median           |
| Total (All Sites) |       920 |                   30 |                3.3 |          172 |       18.7 |                   202 |                22   | Converted to NaN; median 239.0 mg/dL replaces all 202 records                |

### 4. Epidemiological and Methodological Findings

1. **Confounding Between Site and Sex:**
   The clinical cohorts exhibit severe sex distribution imbalances.
   VA Long Beach is an adult Veterans Affairs hospital cohort with 97.0%
   male representation (194/200) and an overall disease rate of 74.5%.
   Switzerland is 91.9% male (113/123) with 93.5% prevalence. In contrast,
   Cleveland (31.9% female) and Hungary (27.6% female) include substantial
   female populations whose disease rates (25.8% and 14.8%) are less than
   half those of males. Consequently, site differences in baseline risk and
   predictive values are partially confounded with sex distribution.

2. **The 202 `chol` Missing Count and the Imputed Constant:**
   Across the 920 patients, 202 records (22.0%) lack serum cholesterol data.
   Critically, 172 of these were encoded as `0 mg/dL` (123 in Switzerland,
   49 in VA Long Beach), while 30 were encoded as explicit `NaN` (23 in
   Hungary, 7 in VA Long Beach). A serum cholesterol of 0 mg/dL is
   physiologically incompatible with human life. In models utilizing cholesterol,
   these 202 values are imputed using the training split median (239.0 mg/dL).
   For the entire Swiss cohort (100.0%) and over a quarter of the VA cohort
   (28.0%), cholesterol is imputed to the identical constant 239.0 mg/dL,
   providing zero variance and zero discriminative signal for those centers.
   Removing cholesterol in Ablation Set D eliminates this uninformative synthetic
   constant without loss of cross-validated discrimination.

3. **Site-Level Prevalence Shifts:**
   True disease prevalence shifts dramatically across centers: Hungary (36.2%),
   Cleveland (46.1%), VA Long Beach (74.5%), and Switzerland (93.5%). Because of
   these baseline shifts, a single universal decision threshold produces varying
   sensitivity and false alarm rates depending on center baseline risk.

4. **Heterogeneous Missingness across Exercise Stress Parameters:**
   While Cleveland has complete records (0% missing across clinical features),
   VA Long Beach exhibits ~26-28% missingness across exercise treadmill
   parameters (`thalach` 26.5%, `oldpeak` 28.0%, `exang` 26.5%, `trestbps`
   28.5%). Switzerland similarly missed fasting blood sugar (`fbs` 61.0%
   missing). This confirms that cross-hospital deployment must tolerate
   selective missingness through leak-free pipeline imputation.
