from pathlib import Path
import pandas as pd # type: ignore
import numpy as np # type: ignore

from merchandising_analysis import MerchandisingAnalyzer
from config import DATE_COL, TIME_COL, MERCH_LEVELS # type: ignore

class StratifiedAnalyzer(MerchandisingAnalyzer):
    """
    Merchandising analysis stratified by month ranges and time slots.
    """

    def __init__(self, df: pd.DataFrame, figures_dir: Path):
        super().__init__(df, figures_dir)
        self._prepare_datetime()

    def _prepare_datetime(self) -> None:
        # Build a datetime column from date + time
        if DATE_COL not in self.df.columns or TIME_COL not in self.df.columns:
            return

        # Build datetime combining date and time
        dt = pd.to_datetime(
            self.df[DATE_COL].dt.strftime("%Y-%m-%d") + " " +
            self.df[TIME_COL].astype(str),
            errors="coerce"
        )
        self.df["datetime"] = dt

    def _add_month_range(self) -> None:
        # Add R1/R2/R3 month ranges
        if DATE_COL not in self.df.columns:
            return

        month = self.df[DATE_COL].dt.month
        day = self.df[DATE_COL].dt.day

        cond1 = (month < 5) | ((month == 5) & (day <= 15))
        cond2 = ((month == 5) & (day > 15)) | ((month > 5) & (month <= 9))
        cond3 = month >= 10

        self.df["month_range"] = pd.Series(index=self.df.index, dtype="string")
        self.df.loc[cond1, "month_range"] = "R1_Jan_midMay"
        self.df.loc[cond2, "month_range"] = "R2_midMay_Sep"
        self.df.loc[cond3, "month_range"] = "R3_Oct_Dec"

    def _add_time_slot(self) -> None:
        # Add time slots based on minutes of day
        if "datetime" not in self.df.columns:
            return

        dt = self.df["datetime"]
        hour_min = dt.dt.hour * 60 + dt.dt.minute

        bins = [0, 8 * 60 + 30, 12 * 60 + 30, 16 * 60 + 30, 20 * 60 + 30, 24 * 60]
        labels = ["before_8_30", "S1_08_30_12_30", "S2_12_30_16_30",
                  "S3_16_30_20_30", "after_20_30"]

        self.df["time_slot"] = pd.cut(hour_min, bins=bins, labels=labels,
                                      right=False, include_lowest=True)

    def run_month_ranges(self) -> None:
        # Run analysis for each month range
        self._add_month_range()
        if "month_range" not in self.df.columns:
            return

        for r in self.df["month_range"].dropna().unique():
            sub = self.df[self.df["month_range"] == r]
            for level in MERCH_LEVELS:
                if level in sub.columns:
                    self._plot_top_bottom(sub[level], level, suffix=f"_{r}")

    def run_time_slots(self) -> None:
        # Run analysis for S1/S2/S3 only
        self._add_time_slot()
        if "time_slot" not in self.df.columns:
            return

        valid_slots = ["S1_08_30_12_30", "S2_12_30_16_30", "S3_16_30_20_30"]
        for s in valid_slots:
            sub = self.df[self.df["time_slot"] == s]
            for level in MERCH_LEVELS:
                if level in sub.columns:
                    self._plot_top_bottom(sub[level], level, suffix=f"_{s}")
