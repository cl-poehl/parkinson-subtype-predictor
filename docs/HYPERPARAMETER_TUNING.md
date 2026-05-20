# Hyperparameter Tuning (Nested CV)

Outer folds: 5, inner folds: 3, Optuna trials per outer fold: 50.

## Tuned outer-fold AUCs per classifier

| Classifier | Mean tuned AUC | SD | Default AUC (from main CV) |
|------------|----------------|----|----------------------------|
| random_forest             | 0.947 | 0.044 | 0.943 |
| xgboost                   | 0.948 | 0.042 | 0.945 |
| logistic_regression       | 0.903 | 0.045 | 0.905 |

## Interpretation

If tuned AUC is not substantially higher than default AUC (within 1 SD), the conclusion is that default hyperparameters are already near-optimal for this cohort size and feature set. This is the expected outcome at n=409 with 34 features.

## Best parameters per outer fold

### random_forest

- Fold 0: {'n_estimators': 800, 'max_depth': 4, 'min_samples_leaf': 1, 'max_features': 'log2'} -> outer AUC 0.927
- Fold 1: {'n_estimators': 800, 'max_depth': 6, 'min_samples_leaf': 17, 'max_features': 'sqrt'} -> outer AUC 0.959
- Fold 2: {'n_estimators': 300, 'max_depth': 20, 'min_samples_leaf': 15, 'max_features': 'sqrt'} -> outer AUC 1.000
- Fold 3: {'n_estimators': 500, 'max_depth': 11, 'min_samples_leaf': 2, 'max_features': 'sqrt'} -> outer AUC 0.884
- Fold 4: {'n_estimators': 400, 'max_depth': 16, 'min_samples_leaf': 5, 'max_features': 0.5} -> outer AUC 0.966

### xgboost

- Fold 0: {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.07239327873175212, 'subsample': 0.5013058775526665, 'colsample_bytree': 0.9820463737955013} -> outer AUC 0.927
- Fold 1: {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.030315725522749647, 'subsample': 0.5089809378074098, 'colsample_bytree': 0.8285618345363206} -> outer AUC 0.965
- Fold 2: {'n_estimators': 200, 'max_depth': 2, 'learning_rate': 0.19030368381735815, 'subsample': 0.8005575058716043, 'colsample_bytree': 0.8540362888980227} -> outer AUC 0.986
- Fold 3: {'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.0286618947980441, 'subsample': 0.5367501368011154, 'colsample_bytree': 0.8377667746000513} -> outer AUC 0.884
- Fold 4: {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.013031219870851108, 'subsample': 0.6061908466229987, 'colsample_bytree': 0.6396777696943234} -> outer AUC 0.977

### logistic_regression

- Fold 0: {'C': 2.871704329664153, 'penalty': 'l1'} -> outer AUC 0.880
- Fold 1: {'C': 0.022590912938207773, 'penalty': 'elasticnet'} -> outer AUC 0.848
- Fold 2: {'C': 5.377461054066278, 'penalty': 'l1'} -> outer AUC 0.967
- Fold 3: {'C': 0.1440391093220845, 'penalty': 'l1'} -> outer AUC 0.901
- Fold 4: {'C': 0.04579588598371666, 'penalty': 'l1'} -> outer AUC 0.921
