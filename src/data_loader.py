import pandas as pd # type: ignore
from pathlib import Path
from typing import Optional
from config import DATE_COL, TIME_COL, DESCR_PROD_COL # type: ignore

class DataLoader:
    """
    Load and preprocess the supermarket fidelity dataset.
    """

    def __init__(self, path: Path, sep: str = ",", decimal: str = "."):
        self.path = path
        self.sep = sep
        self.decimal = decimal
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.path, sep=self.sep, decimal=self.decimal)
        return self.df

    def preprocess(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Call load() before preprocess().")

        df = self.df.copy()

        # Parse date
        if DATE_COL in df.columns:
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%Y-%m-%d",
                                          errors="coerce")
            df = df.dropna(subset=[DATE_COL])

        # Parse time
        if TIME_COL in df.columns:
            df[TIME_COL] = pd.to_datetime(df[TIME_COL],
                                          format="%H:%M",
                                          errors="coerce").dt.time
            df = df.dropna(subset=[TIME_COL])

        # Remove shoppers (non-product items)
        if DESCR_PROD_COL in df.columns:
            df = df[~df[DESCR_PROD_COL]
                    .str.contains("SHOPPER", case=False, na=False)]

        self.df = df
        return self.df
