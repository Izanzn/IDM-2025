from pathlib import Path

# Base paths (relative to project root)
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "AnonymizedFidelity.csv"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

# General parameters
DATE_COL = "data"
TIME_COL = "ora"
RECEIPT_COL = "scontrino_id"
CARD_COL = "tessera"
PRODUCT_COL = "cod_prod"
DESCR_PROD_COL = "descr_prod"

MERCH_LEVELS = ["liv1", "liv2", "liv3", "liv4"]
