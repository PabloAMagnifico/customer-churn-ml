from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_model_specs():
    return {
        "baseline_most_frequent": {
            "estimator": DummyClassifier(strategy="most_frequent"),
            "params": {"strategy": "most_frequent"},
        },
        "baseline_stratified": {
            "estimator": DummyClassifier(strategy="stratified", random_state=42),
            "params": {"strategy": "stratified", "random_state": 42},
        },
        "logistic_c01": {
            "estimator": LogisticRegression(C=0.1, max_iter=2000, solver="liblinear", random_state=42),
            "params": {"C": 0.1, "max_iter": 2000, "solver": "liblinear"},
        },
        "logistic_c1": {
            "estimator": LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=42),
            "params": {"C": 1.0, "max_iter": 2000, "solver": "liblinear"},
        },
        "rf_depth6": {
            "estimator": RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1),
            "params": {"n_estimators": 250, "max_depth": 6, "min_samples_leaf": 3, "random_state": 42},
        },
        "rf_depth12": {
            "estimator": RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1),
            "params": {"n_estimators": 250, "max_depth": 12, "min_samples_leaf": 2, "random_state": 42},
        },
    }
