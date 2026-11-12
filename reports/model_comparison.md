# Model Cross-Validation Comparison Table

Repeated Stratified Cross-Validation (5 folds x 10 repeats = 50 splits) performance of all tuned models on the training set ($N=736$). The ± indicates the sample standard deviation across the 50 fold scores:

| model               | hyperparameters                                                                                                       | accuracy        | precision       | recall          | f1              | roc_auc         | pr_auc          | brier_score     |
|:--------------------|:----------------------------------------------------------------------------------------------------------------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| logistic_regression | C=1.0, max_iter=2000, penalty=l2, solver=lbfgs                                                                        | 0.7961 ± 0.0283 | 0.8049 ± 0.0319 | 0.8356 ± 0.0468 | 0.8189 ± 0.0267 | 0.8685 ± 0.0255 | 0.8764 ± 0.0316 | 0.1453 ± 0.0152 |
| random_forest       | max_depth=5, max_features=log2, min_samples_leaf=1, min_samples_split=10, n_estimators=500                            | 0.8004 ± 0.0271 | 0.7941 ± 0.0303 | 0.8651 ± 0.0426 | 0.8272 ± 0.0246 | 0.8659 ± 0.0269 | 0.8670 ± 0.0333 | 0.1478 ± 0.0123 |
| xgboost             | colsample_bytree=0.6, gamma=0.3, learning_rate=0.01, max_depth=3, min_child_weight=5, n_estimators=300, subsample=1.0 | 0.7954 ± 0.0271 | 0.7946 ± 0.0322 | 0.8523 ± 0.0410 | 0.8215 ± 0.0241 | 0.8648 ± 0.0274 | 0.8661 ± 0.0324 | 0.1481 ± 0.0131 |
