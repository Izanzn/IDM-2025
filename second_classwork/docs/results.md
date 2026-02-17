# Results Report — DAES Dataset (First Classwork)

## 1. Objective
The goal of this classwork is to build a complete machine learning workflow for classifying patients into three groups (ASD, GDD, Controls) using the DAES dataset. The workflow includes preprocessing, PCA visualization, model selection via GridSearchCV with 5-fold stratified cross-validation, and evaluation on a held-out test set. Ensemble methods (bagging and boosting) are included.

## 2. Dataset
The dataset is an Excel workbook with three sheets:
- ASD
- GDD
- Controlli (mapped to Controls)

## 3. Preprocessing
The following preprocessing steps were applied:

1. **Header reconstruction:** the first row of each sheet contains the actual feature identifiers (e.g., B1, D1, ...), therefore headers were reconstructed accordingly.
2. **Filtering:** patients with **Età equivalente < 12** were removed.
3. **Required column removal:** the following fields were removed if present:
   - Età cronologica (mesi), Scala B, Scala D, TOT., Score di rischio
4. **Identifier removal:** Pazienti was removed if present.
5. **Encoding:** Sesso was encoded as M=1, F=0 (missing values filled with the mode).
6. **Missing values:** remaining features were converted to numeric; missing values were imputed using the median (final fallback used to avoid invalid NaNs).

## 4. PCA visualization
A 2D PCA projection was computed on standardized features (StandardScaler).

Explained variance ratio:
- PC1 = 0.23785422
- PC2 = 0.10378951  
Total explained variance (PC1+PC2) ≈ 0.34164373 (≈ 34.16%)

The PCA scatter plot is saved as `pca_2d.png`. Since only ~34% of the variance is captured by the first two components, overlap between classes is expected in 2D.

## 5. Experimental protocol
- **Train/Test split:** 80% / 20%, stratified by class, random_state=42
- **Model selection:** GridSearchCV with 5-fold StratifiedKFold (shuffle=True, random_state=42)
- **Metrics on test set:** Accuracy, Balanced Accuracy, Macro-F1

Macro-F1 is reported to evaluate balanced performance across classes.

## 6. Models
The following classifiers were evaluated:
- Decision Tree
- Random Forest
- SVC
- KNN
- Bagging (Decision Tree base estimator)
- AdaBoost (Decision Tree base estimator)

## 7. Results

### 7.1 Summary table
| Model | Best CV Score | Test Accuracy | Test Balanced Accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|
| AdaBoost_DecisionTree | 0.681653 | 0.650 | 0.651948 | **0.642735** |
| DecisionTree | 0.623992 | 0.650 | 0.650361 | 0.629776 |
| RandomForest | **0.738911** | 0.625 | 0.615152 | 0.618600 |
| Bagging_DecisionTree | 0.707056 | 0.625 | 0.621645 | 0.618455 |
| KNN | 0.598589 | 0.600 | 0.599423 | 0.598629 |
| SVC | 0.694355 | 0.600 | 0.599423 | 0.596720 |

Full outputs are stored in:
- `results.csv`
- `classification_reports.txt`

### 7.2 Selected hyperparameters (best_params)
- AdaBoost_DecisionTree: `{'clf__learning_rate': 0.1, 'clf__n_estimators': 200}`  
- DecisionTree: `{'clf__criterion': 'entropy', 'clf__max_depth': None, 'clf__min_samples_leaf': 2, 'clf__min_samples_split': 10}`  
- RandomForest: `{'clf__max_depth': 10, 'clf__max_features': 'log2', 'clf__min_samples_leaf': 1, 'clf__n_estimators': 200}`  
- Bagging_DecisionTree: `{'clf__max_features': 0.7, 'clf__max_samples': 0.7, 'clf__n_estimators': 200}`  
- KNN: `{'clf__n_neighbors': 5, 'clf__p': 2, 'clf__weights': 'uniform'}`  
- SVC: `{'clf__C': 0.1, 'clf__class_weight': 'balanced', 'clf__gamma': 'scale', 'clf__kernel': 'linear'}`

### 7.3 Discussion
AdaBoost and Decision Tree tie on test accuracy (0.65), but **AdaBoost achieves the best Macro-F1 (0.6427)** and the highest balanced accuracy, indicating the most balanced performance across the three classes. Random Forest achieves the highest CV score (0.7389) but a lower test score on this split, which can occur due to variability in a single hold-out test set.

## 8. Conclusion
A full supervised learning workflow was implemented with preprocessing, PCA visualization, and model selection using 5-fold stratified CV. Under balanced evaluation (Macro-F1), **AdaBoost (Decision Tree base estimator)** achieved the best overall performance on the test split. The PCA results (~34% variance in 2D) suggest class overlap, consistent with moderate classification scores.
``
