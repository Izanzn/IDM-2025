from dataclasses import dataclass
from typing import Dict, Any, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    param_grid: Dict[str, Any]

def get_model_specs(random_state: int = 42) -> List[ModelSpec]:
    specs = []

    specs.append(ModelSpec(
        name="DecisionTree",
        pipeline=Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=random_state)),
        ]),
        param_grid={
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__criterion": ["gini", "entropy"],
        }
    ))

    specs.append(ModelSpec(
        name="RandomForest",
        pipeline=Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=random_state)),
        ]),
        param_grid={
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [None, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", None],
        }
    ))

    specs.append(ModelSpec(
        name="SVC",
        pipeline=Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC()),
        ]),
        param_grid={
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["rbf", "linear"],
            "clf__gamma": ["scale", "auto"],
            "clf__class_weight": [None, "balanced"],
        }
    ))

    specs.append(ModelSpec(
        name="KNN",
        pipeline=Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ]),
        param_grid={
            "clf__n_neighbors": [3, 5, 7, 9],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        }
    ))

    specs.append(ModelSpec(
        name="Bagging_DecisionTree",
        pipeline=Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=random_state),
                random_state=random_state
            )),
        ]),
        param_grid={
            "clf__n_estimators": [50, 100, 200],
            "clf__max_samples": [0.7, 1.0],
            "clf__max_features": [0.7, 1.0],
        }
    ))

    specs.append(ModelSpec(
        name="AdaBoost_DecisionTree",
        pipeline=Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2, random_state=random_state),
                random_state=random_state
            )),
        ]),
        param_grid={
            "clf__n_estimators": [50, 100, 200],
            "clf__learning_rate": [0.1, 0.5, 1.0],
        }
    ))

    return specs
