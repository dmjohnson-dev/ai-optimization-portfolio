import os
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.inspection import permutation_importance

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "DQN1_Dataset.xlsx")
SHEET_NAME = "Data"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "task2")

TARGETS_AIR = ["pm2.5", "no2", "co2"]
TARGET_HEALTH = "healthRiskScore"
SEED = 42


def add_time_features(X: pd.DataFrame, dt: pd.Series) -> pd.DataFrame:
    X = X.copy()
    X["year"] = dt.dt.year
    X["dayofyear"] = dt.dt.dayofyear
    X["month"] = dt.dt.month
    X["day"] = dt.dt.day
    X["hour"] = dt.dt.hour
    return X


def rmse_value(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


rmse_scorer = make_scorer(lambda yt, yp: rmse_value(yt, yp), greater_is_better=False)


def load_split():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    if "datetimeEpoch" not in df.columns:
        raise ValueError("Missing required column: datetimeEpoch")

    df["datetime"] = pd.to_datetime(df["datetimeEpoch"], unit="s", errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    X_air = df.drop(columns=TARGETS_AIR + [TARGET_HEALTH], errors="ignore")
    X_air = X_air.drop(columns=["datetimeEpoch", "datetime"], errors="ignore")
    X_air = add_time_features(X_air, df["datetime"])
    X_air = X_air.select_dtypes(include=[np.number])
    Y_air = df[TARGETS_AIR].copy()

    X_h = df.drop(columns=[TARGET_HEALTH], errors="ignore")
    X_h = X_h.drop(columns=["datetimeEpoch", "datetime"], errors="ignore")
    X_h = add_time_features(X_h, df["datetime"])
    X_h = X_h.select_dtypes(include=[np.number])
    Y_h = df[TARGET_HEALTH].copy()

    split = int(len(df) * 0.8)
    return (
        X_air.iloc[:split], X_air.iloc[split:], Y_air.iloc[:split], Y_air.iloc[split:],
        X_h.iloc[:split], X_h.iloc[split:], Y_h.iloc[:split], Y_h.iloc[split:]
    )


def baseline_models():
    air = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", MultiOutputRegressor(
            RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        ))
    ])
    health = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1))
    ])
    return air, health


# B1
def b1_random_search_air(X_train, Y_train):
    base = MultiOutputRegressor(RandomForestRegressor(random_state=SEED, n_jobs=-1))
    tscv = TimeSeriesSplit(n_splits=5)
    param_dist = {
        "estimator__n_estimators": [200, 300, 400, 600],
        "estimator__max_depth": [None, 10, 15, 25],
        "estimator__min_samples_leaf": [1, 2, 4, 6],
        "estimator__min_samples_split": [2, 5, 10],
        "estimator__max_features": ["sqrt", 0.5, 0.8],
    }
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=20,
        scoring=rmse_scorer,
        cv=tscv,
        random_state=SEED,
        n_jobs=-1,
    )
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("search", search)])
    pipe.fit(X_train, Y_train)
    return pipe.named_steps["search"].best_params_, pipe.named_steps["search"].best_score_


def b1_random_search_health(X_train, y_train):
    base = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=5)
    param_dist = {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [None, 10, 15, 25],
        "min_samples_leaf": [1, 2, 4, 6],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", 0.5, 0.8],
    }
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=20,
        scoring=rmse_scorer,
        cv=tscv,
        random_state=SEED,
        n_jobs=-1,
    )
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("search", search)])
    pipe.fit(X_train, y_train)
    return pipe.named_steps["search"].best_params_, pipe.named_steps["search"].best_score_


def b1_feature_selection_health(fitted_health_pipeline, X_train, y_train, top_k=15):
    imputer = fitted_health_pipeline.named_steps["imputer"]
    model = fitted_health_pipeline.named_steps["rf"]
    X_imp = imputer.transform(X_train)
    result = permutation_importance(
        model, X_imp, y_train,
        n_repeats=5, random_state=SEED, n_jobs=-1
    )
    feature_names = X_train.columns.tolist()
    ranked = sorted(zip(feature_names, result.importances_mean), key=lambda x: x[1], reverse=True)
    keep = [name for name, _ in ranked[:top_k]]
    return keep, ranked


# B2
def b2_regularized_health_models():
    reg_constraints = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=500, random_state=SEED, n_jobs=-1,
            max_depth=12, min_samples_leaf=5, min_samples_split=10
        ))
    ])
    reg_pruning = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=500, random_state=SEED, n_jobs=-1,
            ccp_alpha=0.0005, min_samples_leaf=2
        ))
    ])
    return reg_constraints, reg_pruning


# -----------------------------
# B3 Ensemble Learning (NEW)
# -----------------------------
def b3_ensemble_models():
    """
    Two ensemble techniques:
    1) ExtraTreesRegressor (different randomized tree ensemble)
    2) VotingRegressor (averaging predictions from multiple models)
    """
    extra = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(n_estimators=600, random_state=SEED, n_jobs=-1))
    ])

    rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
    et = ExtraTreesRegressor(n_estimators=600, random_state=SEED, n_jobs=-1)

    vote = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", VotingRegressor([("rf", rf), ("et", et)]))
    ])

    return extra, vote


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    (X_air_tr, X_air_te, Y_air_tr, Y_air_te,
     X_h_tr, X_h_te, Y_h_tr, Y_h_te) = load_split()

    # Baseline fit (for B1 feature selection)
    _, health_base = baseline_models()
    health_base.fit(X_h_tr, Y_h_tr)

    # B1 artifacts
    air_best_params, air_best_score = b1_random_search_air(X_air_tr, Y_air_tr)
    h_best_params, h_best_score = b1_random_search_health(X_h_tr, Y_h_tr)
    keep_feats, ranked = b1_feature_selection_health(health_base, X_h_tr, Y_h_tr, top_k=15)

    with open(os.path.join(OUT_DIR, "b1_best_params.json"), "w", encoding="utf-8") as f:
        json.dump({
            "air_best_params": air_best_params,
            "air_cv_score_neg_rmse": air_best_score,
            "health_best_params": h_best_params,
            "health_cv_score_neg_rmse": h_best_score
        }, f, indent=2)

    pd.DataFrame(ranked, columns=["feature", "perm_importance_mean"]).to_csv(
        os.path.join(OUT_DIR, "b1_health_feature_importance.csv"), index=False
    )
    pd.DataFrame({"selected_features": keep_feats}).to_csv(
        os.path.join(OUT_DIR, "b1_health_selected_features.csv"), index=False
    )

    # B2 apply regularization
    reg_constraints, reg_pruning = b2_regularized_health_models()
    reg_constraints.fit(X_h_tr, Y_h_tr)
    reg_pruning.fit(X_h_tr, Y_h_tr)

    with open(os.path.join(OUT_DIR, "b2_regularization_settings.json"), "w", encoding="utf-8") as f:
        json.dump({
            "reg_constraints": {"max_depth": 12, "min_samples_leaf": 5, "min_samples_split": 10},
            "reg_pruning": {"ccp_alpha": 0.0005, "min_samples_leaf": 2}
        }, f, indent=2)

    # B3 apply ensemble learning
    extra, vote = b3_ensemble_models()
    extra.fit(X_h_tr, Y_h_tr)
    vote.fit(X_h_tr, Y_h_tr)

    with open(os.path.join(OUT_DIR, "b3_ensemble_settings.json"), "w", encoding="utf-8") as f:
        json.dump({
            "ensemble_1": "ExtraTreesRegressor(n_estimators=600)",
            "ensemble_2": "VotingRegressor(RandomForestRegressor + ExtraTreesRegressor)"
        }, f, indent=2)

    print("B3 complete:")
    print("- Saved ensemble settings: outputs/task2/b3_ensemble_settings.json")


if __name__ == "__main__":
    main()
