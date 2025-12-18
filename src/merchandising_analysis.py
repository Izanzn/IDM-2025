from pathlib import Path
from typing import List
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore

from config import MERCH_LEVELS # type: ignore

class MerchandisingAnalyzer:
    """
    Basic merchandising frequency analysis for liv1-liv4.
    """

    def __init__(self, df: pd.DataFrame, figures_dir: Path):
        self.df = df
        self.figures_dir = figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def _plot_top_bottom(self, series: pd.Series,
                         level_name: str,
                         suffix: str = "") -> None:
        # Count category frequencies
        counts = series.value_counts()

        if counts.empty:
            return

        top5 = counts.head(5)
        bottom5 = counts.tail(5)

        # Top 5
        plt.figure()
        top5.plot(kind="bar")
        plt.title(f"Top 5 {level_name} {suffix}")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"{level_name}_top5{suffix}.png")
        plt.close()

        # Bottom 5
        plt.figure()
        bottom5.plot(kind="bar")
        plt.title(f"Bottom 5 {level_name} {suffix}")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"{level_name}_bottom5{suffix}.png")
        plt.close()

    def run_task1(self, levels: List[str] = None) -> None:
        if levels is None:
            levels = MERCH_LEVELS

        for level in levels:
            if level in self.df.columns:
                self._plot_top_bottom(self.df[level], level)
