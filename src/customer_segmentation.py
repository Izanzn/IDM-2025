import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.metrics import silhouette_score  # type: ignore

from config import CARD_COL, PRODUCT_COL # type: ignore


class CustomerSegmentation:
    
    """
    Customer segmentation using tessera x product matrix,
    PCA and K-means clustering.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        product_col: str = PRODUCT_COL,
        card_col: str = CARD_COL,
    ):
        # Keep only rows with a valid loyalty card id
        self.df = df[(df[card_col].notna()) & (df[card_col] != "")]
        self.product_col = product_col
        self.card_col = card_col

        self.card_index = None
        self.pca: PCA | None = None
        self.kmeans: KMeans | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def build_card_product_matrix(
        self,
        top_n_products: int | None = 200,
    ) -> pd.DataFrame:
        # Build customer (card) x product matrix using total purchased quantity
        if self.card_col not in self.df.columns:
            raise ValueError(f"Missing card column: {self.card_col}")
        if self.product_col not in self.df.columns:
            raise ValueError(f"Missing product column: {self.product_col}")

        df = self.df

        # Optionally keep only the top-N most sold products to reduce dimensionality
        if top_n_products is not None:
            product_totals = (
                df.groupby(self.product_col)["r_qta_pezzi"]
                .sum()
                .sort_values(ascending=False)
            )
            top_products = product_totals.head(top_n_products).index
            df = df[df[self.product_col].isin(top_products)]

        mat = (
            df.groupby([self.card_col, self.product_col])["r_qta_pezzi"]
            .sum()
            .unstack()
            .fillna(0)
        )
        return mat

    def run_pca(
        self,
        n_components: int = 10,
        top_n_products: int | None = 200,
    ) -> np.ndarray:
        # Standardize the matrix and project customers into PCA space
        mat = self.build_card_product_matrix(top_n_products=top_n_products)
        self.card_index = mat.index

        scaler = StandardScaler(with_mean=True, with_std=True)
        mat_scaled = scaler.fit_transform(mat)

        # Do not ask for more components than available features
        n_comp_eff = min(n_components, mat_scaled.shape[1])

        pca = PCA(n_components=n_comp_eff)
        coords = pca.fit_transform(mat_scaled)

        self.pca = pca
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        return coords

    def choose_k_by_silhouette(
        self,
        ks=range(2, 9),
        n_components: int = 10,
        sample_size: int = 3000,
        top_n_products: int | None = 200,
    ):
        # Choose the best k using silhouette score on a (possibly) sampled set of customers
        mat = self.build_card_product_matrix(top_n_products=top_n_products)

        # Sample customers to speed up silhouette evaluation
        if mat.shape[0] > sample_size:
            rng = np.random.default_rng(42)
            sampled_index = rng.choice(mat.index, size=sample_size, replace=False)
            mat_sample = mat.loc[sampled_index]
        else:
            mat_sample = mat

        scaler = StandardScaler(with_mean=True, with_std=True)
        mat_scaled = scaler.fit_transform(mat_sample)

        n_comp_eff = min(n_components, mat_scaled.shape[1])
        pca = PCA(n_components=n_comp_eff)
        coords = pca.fit_transform(mat_scaled)

        best_k = None
        best_score = -1.0
        scores: dict[int, float] = {}

        for k in ks:
            # k must be smaller than number of points
            if k >= coords.shape[0]:
                continue

            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(coords)
            score = silhouette_score(coords, labels)
            scores[k] = score

            if score > best_score:
                best_score = score
                best_k = k

        return best_k, best_score, scores

    def cluster_cards(
        self,
        n_clusters: int | None = None,
        n_components: int = 10,
        ks=range(2, 9),
        sample_size: int = 3000,
        top_n_products: int | None = 200,
    ) -> pd.DataFrame:
        # Cluster all customers; if k is not provided, select it via silhouette
        if n_clusters is None:
            best_k, best_score, scores = self.choose_k_by_silhouette(
                ks=ks,
                n_components=n_components,
                sample_size=sample_size,
                top_n_products=top_n_products,
            )
            if best_k is None:
                raise RuntimeError("Could not determine a valid k using silhouette.")

            print(f"[INFO] Silhouette scores by k: {scores}")
            print(f"[INFO] Selected k={best_k} by silhouette (score={best_score:.3f}).")
            n_clusters = best_k

        coords = self.run_pca(
            n_components=n_components,
            top_n_products=top_n_products,
        )

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = km.fit_predict(coords)

        self.kmeans = km

        return pd.DataFrame({
            self.card_col: self.card_index,
            "cluster": labels,
        })
