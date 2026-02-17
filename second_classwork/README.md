# Second Classwork — DAES Dataset (IDM 2025)

This project implements the required pipeline for the DAES dataset: preprocessing, 2D PCA visualization, and supervised classification with hyperparameter tuning (GridSearchCV + 5-fold Stratified CV), including bagging and boosting.

## Project structure

first_classwork/
- README.md
- src/
  - data.py            # load + preprocess Excel sheets
  - models.py          # pipelines + parameter grids
  - experiments.py     # GridSearchCV, evaluation, metrics
  - run.py             # runs PCA + training and saves outputs
- docs/
  - results.md         # final report (method + results)
  - pca_2d.png
  - pca_info.txt
  - results.csv
  - classification_reports.txt

## Requirements

Python 3.10+  
Packages: pandas, numpy, scikit-learn, matplotlib, openpyxl

Install:
```bash
pip install -U pandas numpy scikit-learn matplotlib openpyxl
