import numpy as np
import pandas as pd
from typing import Tuple

REMOVE_COLS = [
    "Età cronologica (mesi)",
    "Scala B",
    "Scala D",
    "TOT.",
    "Score di rischio",
]

SHEET_TO_CLASS = {
    "ASD": "ASD",
    "GDD": "GDD",
    "Controlli": "Controls",
}

def _make_unique(cols):
    seen = {}
    out = []
    for c in cols:
        c = str(c).strip()
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
    return out

def _clean_sheet(df: pd.DataFrame) -> pd.DataFrame:
    # Row 0 contains the actual feature codes (B1, D1, etc.)
    header = df.iloc[0]
    new_cols = []
    for col, h in zip(df.columns, header):
        if pd.isna(h):
            new_cols.append(str(col).strip())
        else:
            new_cols.append(str(h).strip())

    out = df.iloc[1:].copy()
    out.columns = _make_unique(new_cols)
    return out

def load_dataset(excel_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    frames = []
    for sheet, label in SHEET_TO_CLASS.items():
        df = pd.read_excel(excel_path, sheet_name=sheet)
        df = _clean_sheet(df)
        df["Class"] = label
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # Validate required column
    if "Età equivalente" not in data.columns:
        raise KeyError("Missing required column: 'Età equivalente'")

    # Filter by equivalent age
    data["Età equivalente"] = pd.to_numeric(data["Età equivalente"], errors="coerce")
    data = data[data["Età equivalente"] >= 12].copy()

    # Remove forbidden columns
    to_drop = [c for c in REMOVE_COLS if c in data.columns]
    if to_drop:
        data.drop(columns=to_drop, inplace=True)

    # Remove patient id if present
    if "Pazienti" in data.columns:
        data.drop(columns="Pazienti", inplace=True)

    # Encode sex if present
    if "Sesso" in data.columns:
        s = data["Sesso"].astype(str).str.strip().str.upper()
        data["Sesso"] = s.map({"M": 1, "F": 0})
        if data["Sesso"].isna().any():
            # if mode is empty (rare), fallback to 0
            mode_vals = data["Sesso"].mode()
            fill_val = int(mode_vals.iloc[0]) if len(mode_vals) else 0
            data["Sesso"] = data["Sesso"].fillna(fill_val)

    # Split X/y
    if "Class" not in data.columns:
        raise KeyError("Missing target column: 'Class'")

    y = data["Class"]
    X = data.drop(columns="Class")

    # Convert all to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop fully-empty columns (important for robustness)
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
    # NOTE: Missing-value imputation is handled in the sklearn Pipeline
    # (SimpleImputer fitted on the training split) to avoid data leakage.

    return X, y
