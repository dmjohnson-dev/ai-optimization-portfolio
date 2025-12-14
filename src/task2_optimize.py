import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, make_scorer
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


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape_percent(y_true, y_pred) -> float:
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)


rmse_scorer = make_scorer(lambda yt, yp: rmse(yt, yp), greater_is_better=False)


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


def baseline_air():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", MultiOutputRegressor(
            RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        ))
    ])


def baseline_health():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1))
    ])


# B1(1) tuned models
def tuned_air(X_train, Y_train):
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
        base, param_distributions=param_dist, n_iter=20, scoring=rmse_scorer,
        cv=tscv, random_state=SEED, n_jobs=-1
    )
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", search)])
    pipe.fit(X_train, Y_train)
    return pipe


def tuned_health(X_train, y_train):
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
        base, param_distributions=param_dist, n_iter=20, scoring=rmse_scorer,
        cv=tscv, random_state=SEED, n_jobs=-1
    )
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", search)])
    pipe.fit(X_train, y_train)
    return pipe


# B1(2) feature selection (health only)
def feature_select_health(fitted_baseline_health, X_train, y_train, top_k=15):
    imputer = fitted_baseline_health.named_steps["imputer"]
    model = fitted_baseline_health.named_steps["model"]
    X_imp = imputer.transform(X_train)

    res = permutation_importance(model, X_imp, y_train, n_repeats=5, random_state=SEED, n_jobs=-1)
    ranked = sorted(zip(X_train.columns.tolist(), res.importances_mean), key=lambda x: x[1], reverse=True)
    keep = [name for name, _ in ranked[:top_k]]
    return keep


# B2 regularization (health)
def reg_constraints_health():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=500, random_state=SEED, n_jobs=-1,
            max_depth=12, min_samples_leaf=5, min_samples_split=10
        ))
    ])


def reg_pruning_health():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=500, random_state=SEED, n_jobs=-1,
            ccp_alpha=0.0005, min_samples_leaf=2
        ))
    ])


# B3 ensemble (health)
def ensemble_extratrees_health():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(n_estimators=600, random_state=SEED, n_jobs=-1))
    ])


def ensemble_voting_health():
    rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
    et = ExtraTreesRegressor(n_estimators=600, random_state=SEED, n_jobs=-1)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", VotingRegressor([("rf", rf), ("et", et)]))
    ])


def eval_air(Y_true: pd.DataFrame, Y_pred: np.ndarray, label: str) -> list[dict]:
    rows = []
    for i, t in enumerate(TARGETS_AIR):
        rows.append({
            "variant": label,
            "target": t,
            "RMSE": rmse(Y_true[t].values, Y_pred[:, i]),
            "MAPE_percent": mape_percent(Y_true[t].values, Y_pred[:, i]),
        })
    return rows


def eval_health(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    return {
        "variant": label,
        "target": TARGET_HEALTH,
        "RMSE": rmse(y_true, y_pred),
        "MAPE_percent": mape_percent(y_true, y_pred),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    (X_air_tr, X_air_te, Y_air_tr, Y_air_te,
     X_h_tr, X_h_te, Y_h_tr, Y_h_te) = load_split()

    metrics_air = []
    metrics_health = []

    # BASELINE
    air0 = baseline_air()
    h0 = baseline_health()
    air0.fit(X_air_tr, Y_air_tr)
    h0.fit(X_h_tr, Y_h_tr)

    air_pred0 = air0.predict(X_air_te)
    h_pred0 = h0.predict(X_h_te)

    metrics_air += eval_air(Y_air_te, air_pred0, "baseline")
    metrics_health.append(eval_health(Y_h_te.values, h_pred0, "baseline"))

    # B1(1) tuned
    air_t = tuned_air(X_air_tr, Y_air_tr)
    h_t = tuned_health(X_h_tr, Y_h_tr)
    air_pred_t = air_t.predict(X_air_te)
    h_pred_t = h_t.predict(X_h_te)

    metrics_air += eval_air(Y_air_te, air_pred_t, "opt_random_search")
    metrics_health.append(eval_health(Y_h_te.values, h_pred_t, "opt_random_search"))

    # B1(2) feature selection health
    keep = feature_select_health(h0, X_h_tr, Y_h_tr, top_k=15)
    X_h_tr_fs = X_h_tr[keep]
    X_h_te_fs = X_h_te[keep]
    h_fs = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1))
    ])
    h_fs.fit(X_h_tr_fs, Y_h_tr)
    h_pred_fs = h_fs.predict(X_h_te_fs)
    metrics_health.append(eval_health(Y_h_te.values, h_pred_fs, "opt_feature_selection"))

    # B2 regularization
    h_reg1 = reg_constraints_health()
    h_reg2 = reg_pruning_health()
    h_reg1.fit(X_h_tr, Y_h_tr)
    h_reg2.fit(X_h_tr, Y_h_tr)
    metrics_health.append(eval_health(Y_h_te.values, h_reg1.predict(X_h_te), "reg_constraints"))
    metrics_health.append(eval_health(Y_h_te.values, h_reg2.predict(X_h_te), "reg_pruning_ccp_alpha"))

    # B3 ensemble
    h_extra = ensemble_extratrees_health()
    h_vote = ensemble_voting_health()
    h_extra.fit(X_h_tr, Y_h_tr)
    h_vote.fit(X_h_tr, Y_h_tr)
    metrics_health.append(eval_health(Y_h_te.values, h_extra.predict(X_h_te), "ensemble_extratrees"))
    metrics_health.append(eval_health(Y_h_te.values, h_vote.predict(X_h_te), "ensemble_voting"))

    pd.DataFrame(metrics_air).to_csv(os.path.join(OUT_DIR, "task2_air_metrics.csv"), index=False)
    pd.DataFrame(metrics_health).to_csv(os.path.join(OUT_DIR, "task2_health_metrics.csv"), index=False)

    print("C1 complete: saved RMSE + MAPE metrics to outputs/task2/.")


if __name__ == "__main__":
    main()
