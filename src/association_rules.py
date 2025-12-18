import pandas as pd # type: ignore
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules # type: ignore


class AssociationRuleMiner:
    """
    Frequent itemset mining and association rules (Task 3 & 4).

    - Transaction: one receipt (scontrino_id)
    - Item: merchandising category at level liv4 (or another column)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        level_col: str = "liv4",
        id_col: str = "scontrino_id",
    ):
        self.df = df.copy()
        self.level_col = level_col
        self.id_col = id_col
        self.n_transactions: int = 0

    def _build_transaction_matrix(
        self,
        min_support_singleton: float | None = None,
    ) -> pd.DataFrame:
        """
        Build a transaction x item matrix (bool).
        Optionally drop items whose *individual* support
        is below min_support_singleton.
        """
        if self.id_col not in self.df.columns:
            raise ValueError(f"Missing receipt column: {self.id_col}")
        if self.level_col not in self.df.columns:
            raise ValueError(
                f"Missing merchandising level column: {self.level_col}"
            )

        # Count occurrences per (transaction, item)
        basket = (
            self.df
            .groupby([self.id_col, self.level_col])
            .size()
            .unstack()
            .fillna(0)
        )

        # Convert to bool
        basket = basket.astype(bool)

        # Number of transactions
        self.n_transactions = basket.shape[0]

        # Pre-filter items with low singleton support to reduce memory/time
        if min_support_singleton is not None:
            col_support = basket.sum(axis=0) / self.n_transactions
            cols_to_keep = col_support[col_support >= min_support_singleton].index

            # If we drop items with support < min_support, we are NOT losing
            # any potentially frequent itemset, by Apriori principle.
            basket = basket.loc[:, cols_to_keep]

        return basket

    def _postprocess_rules(self, rules: pd.DataFrame) -> pd.DataFrame:

        # Add absolute support/coverage and sort by lift
        if rules.empty:
            return rules

        rules = rules.copy()
        rules["support_abs"] = (rules["support"] * self.n_transactions).astype(int)
        rules["coverage"] = rules["antecedent support"]
        rules["coverage_abs"] = (rules["coverage"] * self.n_transactions).astype(int)
        rules = rules.sort_values("lift", ascending=False)
        return rules

    def run_apriori(
        self,
        min_support: float = 0.01,
        metric: str = "lift",
        min_threshold: float = 1.0,
    ) -> pd.DataFrame:
        
        # Run Apriori on the basket matrix and generate association rules
        basket = self._build_transaction_matrix(
            min_support_singleton=min_support
        )

        if basket.shape[1] == 0:
            # No items with support >= min_support
            return pd.DataFrame()

        freq_items = apriori(
            basket,
            min_support=min_support,
            use_colnames=True,
        )
        if freq_items.empty:
            return pd.DataFrame()

        rules = association_rules(
            freq_items,
            metric=metric,
            min_threshold=min_threshold,
        )
        return self._postprocess_rules(rules)

    def run_fpgrowth(
        self,
        min_support: float = 0.01,
        metric: str = "lift",
        min_threshold: float = 1.0,
    ) -> pd.DataFrame:
        
        # Run FP-Growth on the basket matrix and generate association rules
        basket = self._build_transaction_matrix(
            min_support_singleton=min_support
        )

        if basket.shape[1] == 0:
            return pd.DataFrame()

        freq_items = fpgrowth(
            basket,
            min_support=min_support,
            use_colnames=True,
        )
        if freq_items.empty:
            return pd.DataFrame()

        rules = association_rules(
            freq_items,
            metric=metric,
            min_threshold=min_threshold,
        )
        return self._postprocess_rules(rules)
