from pathlib import Path
import matplotlib.pyplot as plt

from data import load_dataset
from models import get_model_specs
from experiments import ExperimentRunner

def main():
    base_dir = Path(__file__).resolve().parent
    docs_dir = base_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    excel_path = base_dir / "Dataset DAES.xlsx"
    X, y = load_dataset(str(excel_path))

    runner = ExperimentRunner(random_state=42)

    # 1) PCA 2D scatter
    Z, var_ratio = runner.pca_2d(X)
    print("PCA explained variance ratio:", var_ratio)

    plt.figure()
    for cls in sorted(y.unique()):
        idx = (y == cls).values
        plt.scatter(Z[idx, 0], Z[idx, 1], label=cls)
    plt.legend()
    plt.title("PCA (2D) - Patients")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(docs_dir / "pca_2d.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save PCA info for the report
    with open(docs_dir / "pca_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Explained variance ratio (PC1, PC2): {var_ratio.tolist()}\n")
        f.write(f"Total explained variance (PC1+PC2): {float(var_ratio.sum()):.6f}\n")

    # 2-4) GridSearchCV + models + bagging/boosting
    specs = get_model_specs(random_state=42)
    table, reports, _ = runner.run_all(specs, X, y)

    print("\n=== Summary ===")
    print(table.to_string(index=False))

    # Save outputs
    table.to_csv(docs_dir / "results.csv", index=False)

    with open(docs_dir / "classification_reports.txt", "w", encoding="utf-8") as f:
        for name in table["model"].tolist():  # order reports by the final ranking table
            f.write(f"## {name}\n{reports[name]}\n\n")

if __name__ == "__main__":
    main()
