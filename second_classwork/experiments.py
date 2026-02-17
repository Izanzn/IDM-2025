import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, balanced_accuracy_score, confusion_matrix
)

class ExperimentRunner:
    def __init__(self, random_state: int = 42, scoring: str = "accuracy"):
        self.random_state = random_state
        self.scoring = scoring
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    def pca_2d(self, X):
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=self.random_state)
        Z = pca.fit_transform(Xs)
        return Z, pca.explained_variance_ratio_

    def train_test(self, X, y, test_size=0.2):
        return train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

    def grid_search(self, model_spec, X_train, y_train):
        gs = GridSearchCV(
            model_spec.pipeline,
            model_spec.param_grid,
            cv=self.cv,
            n_jobs=-1,
            scoring=self.scoring
        )
        gs.fit(X_train, y_train)
        return gs

    def evaluate(self, estimator, X_test, y_test):
        pred = estimator.predict(X_test)
        labels = sorted(y_test.unique())

        return {
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "f1_macro": f1_score(y_test, pred, average="macro"),
            "confusion": confusion_matrix(y_test, pred, labels=labels),
            "report": classification_report(y_test, pred, zero_division=0),
        }

    def run_all(self, specs, X, y):
        X_train, X_test, y_train, y_test = self.train_test(X, y)

        rows = []
        reports = {}
        confusions = {}

        for spec in specs:
            gs = self.grid_search(spec, X_train, y_train)
            metrics = self.evaluate(gs.best_estimator_, X_test, y_test)

            rows.append({
                "model": spec.name,
                "best_cv_score": gs.best_score_,
                "test_accuracy": metrics["accuracy"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "test_f1_macro": metrics["f1_macro"],
                "best_params": gs.best_params_,
            })
            reports[spec.name] = metrics["report"]
            confusions[spec.name] = metrics["confusion"]

        df = pd.DataFrame(rows).sort_values("test_f1_macro", ascending=False)
        return df, reports, confusions
