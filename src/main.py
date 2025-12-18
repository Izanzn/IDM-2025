from config import DATA_PATH, FIGURES_DIR, RESULTS_DIR # type: ignore
from data_loader import DataLoader
from stratified_analysis import StratifiedAnalyzer
from association_rules import AssociationRuleMiner
from customer_segmentation import CustomerSegmentation
import numpy as np # type: ignore


def main():
    # Create output directories if they do not exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1) Load and preprocess data
    # --------------------------------------------------
    loader = DataLoader(DATA_PATH)
    df = loader.load()
    df = loader.preprocess()

    print(f"[INFO] Dataset after preprocessing: {df.shape[0]} rows, {df.shape[1]} columns")

    # --------------------------------------------------
    # 2) Task 1 & Task 2: Merchandising analysis (global + stratified)
    # --------------------------------------------------
    print("[INFO] Running merchandising analysis (Task 1 & 2)...")
    strat = StratifiedAnalyzer(df, figures_dir=FIGURES_DIR)
    # Task 1: global top/bottom categories for liv1–liv4
    strat.run_task1()
    # Task 2: stratification by month ranges (R1/R2/R3)
    strat.run_month_ranges()
    # Task 2: stratification by time slots (S1/S2/S3)
    strat.run_time_slots()
    print("[INFO] Merchandising analysis done. Figures saved in '../figures/'.")

    # --------------------------------------------------
    # 3) Task 3 & Task 4: Association rules (Apriori + FP-Growth)
    # --------------------------------------------------
    print("[INFO] Preparing data for association rules (Tasks 3 & 4)...")

    if "scontrino_id" not in df.columns:
        raise ValueError("Column 'scontrino_id' not found in dataframe!")

    unique_receipts = df["scontrino_id"].dropna().unique()
    n_receipts = len(unique_receipts)
    print(f"[INFO] Number of distinct receipts: {n_receipts}")

    # Sampling to keep runtime/memory under control
    max_receipts = 4000        # <= puedes subirlo/bajarlo
    min_support_rules = 0.04   # 4% de soporte mínimo

    if n_receipts > max_receipts:
        rng = np.random.default_rng(42)
        sampled_receipts = rng.choice(
            unique_receipts, size=max_receipts, replace=False
        )
        df_rules = df[df["scontrino_id"].isin(sampled_receipts)]
        print(f"[INFO] Using a sample of {max_receipts} receipts for association rules.")
    else:
        df_rules = df
        print("[INFO] Using all receipts for association rules (dataset is small enough).")

    print(f"[INFO] Using min_support={min_support_rules:.2f} for association rules.")

    print("[INFO] Mining association rules with Apriori...")
    miner = AssociationRuleMiner(df_rules, level_col="liv4", id_col="scontrino_id")
    rules_apriori = miner.run_apriori(
        min_support=min_support_rules,
        metric="lift",
        min_threshold=1.0,
    )
    rules_apriori.to_csv(RESULTS_DIR / "rules_apriori.csv", index=False)
    print(f"[INFO] Apriori done. {len(rules_apriori)} rules saved to '../results/rules_apriori.csv'.")

    print("[INFO] Mining association rules with FP-Growth...")
    rules_fpgrowth = miner.run_fpgrowth(
        min_support=min_support_rules,
        metric="lift",
        min_threshold=1.0,
    )
    rules_fpgrowth.to_csv(RESULTS_DIR / "rules_fpgrowth.csv", index=False)
    print(f"[INFO] FP-Growth done. {len(rules_fpgrowth)} rules saved to '../results/rules_fpgrowth.csv'.")

    # --------------------------------------------------
    # 4) Task 5: Customer segmentation (PCA + K-means + silhouette)
    # --------------------------------------------------
    print("[INFO] Running customer segmentation (Task 5)...")
    segmenter = CustomerSegmentation(df)

    # PCA + silhouette parameters (balanced for speed and quality)
    n_components = 5 
    ks = range(2, 8) 
    sample_size = 2000
    top_n_products = 200

    # n_clusters=None -> automatically selected using silhouette
    clusters = segmenter.cluster_cards(
        n_clusters=None,
        n_components=n_components,
        ks=ks,
        sample_size=sample_size,
        top_n_products=top_n_products,
    )
    clusters.to_csv(RESULTS_DIR / "card_clusters.csv", index=False)
    print(f"[INFO] Customer segmentation done. Results saved to '../results/card_clusters.csv'.")

    print("[INFO] Analysis completed. Check '../figures/' and '../results/'.")    


if __name__ == "__main__":
    main()
