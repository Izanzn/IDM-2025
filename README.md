# IDM 2025 – First Classwork

This repository contains the implementation of the **First Classwork**, based on the **AnonymizedFidelity** supermarket dataset.

The solution covers:

- **Task 1**: Global merchandising analysis (Top/Bottom categories) for `liv1–liv4`
- **Task 2**: Stratified merchandising analysis:
  - Month ranges (R1/R2/R3)
  - Time slots (S1/S2/S3)
- **Task 3**: Association rules with **Apriori**
- **Task 4**: Association rules with **FP-Growth**
- **Task 5**: Customer segmentation using **card × product matrix**, **PCA**, **K-means**, and **k selection via silhouette**

---

## Project structure

```text
.
├── data/
│   └── AnonymizedFidelity.csv
├── figures/
│   └── *.png
├── results/
│   ├── rules_apriori.csv
│   ├── rules_fpgrowth.csv
│   └── card_clusters.csv
└── src/
    ├── config.py
    ├── data_loader.py
    ├── merchandising_analysis.py
    ├── stratified_analysis.py
    ├── association_rules.py
    ├── customer_segmentation.py
    └── main.py
```

---

## Requirements

- Python 3.10+
- Packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `mlxtend`

Install:

```bash
pip install pandas numpy matplotlib scikit-learn mlxtend
```

---

## How to run

Run the full pipeline:

```bash
cd src
python main.py
```

Outputs:
- Plots are saved to `figures/`
- CSV results are saved to `results/`

---

## Notes on generated plots

The merchandising pipeline produces **56 plots**:
- Task 1: 4 levels × (Top 5 + Bottom 5) = 8
- Task 2 month ranges: 3 ranges × 4 levels × 2 = 24
- Task 2 time slots: 3 slots × 4 levels × 2 = 24  
Total = 56 plots.

---

## Results obtained

### Association rules (Tasks 3–4)

- Apriori rules: **34**
- FP-Growth rules: **34**
- Apriori and FP-Growth produced the **same rule set**: **True**

Rules are mined at `liv4` level using:
- a random sample of receipts (to keep memory/time manageable)
- `min_support = 0.04`
- rule metric: `lift` (with `min_threshold = 1.0`)

**Top rules by lift (from `results/rules_apriori.csv`):**

| antecedent | consequent | support | confidence | lift | support_abs |
| --- | --- | --- | --- | --- | --- |
| 1150201 | 1150202 | 0.0470 | 0.3456 | 4.1141 | 188 |
| 1150202 | 1150201 | 0.0470 | 0.5595 | 4.1141 | 188 |
| 9010102 | 9010101 | 0.0525 | 0.5172 | 2.9727 | 210 |
| 9010101 | 9010102 | 0.0525 | 0.3017 | 2.9727 | 210 |
| 3060402 | 3060404 | 0.0418 | 0.1960 | 2.9698 | 167 |
| 3060404 | 3060402 | 0.0418 | 0.6326 | 2.9698 | 167 |
| 3060405 | 3060402 | 0.0493 | 0.5760 | 2.7043 | 197 |
| 3060402 | 3060405 | 0.0493 | 0.2312 | 2.7043 | 197 |

> The item identifiers correspond to `liv4` categories (codes). They can be mapped to the original labels in the dataset if needed.

### Customer segmentation (Task 5)

The clustering result is stored in `results/card_clusters.csv` with columns:
- `tessera` (loyalty card id)
- `cluster` (assigned cluster label)

**Number of clustered cards:** 8,486

**Cluster distribution:**
- Cluster 0: 7,514 cards (88.55%)
- Cluster 1: 972 cards (11.45%)

---
