"""
D682 DQN1 Task 1 — Random Forest Solution (Reproducible)

What this script does:
1) Loads the provided Excel dataset (data/DQN1_Dataset.xlsx, sheet "Data")
2) Builds two Random Forest models:
   - Multi-output regression for air quality targets: pm2.5, no2, co2
   - Single-output regression for health risk target: healthRiskScore
3) Uses a time-based 80/20 split
4) Evaluates with two metrics: RMSE and MAPE (%)
5) Saves results to outputs/:
   - air_quality_metrics.csv, health_risk_metrics.csv
   - air_quality_predictions.csv, health_risk_predictions.csv
   - trend_pm2.5.png, trend_no2.png, trend_co2.png, trend_healthRiskScore.png
"""

import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

import matplotlib.pyplot as plt


# -----------------------------
# Paths (robust for PyCharm)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DATA_PATH = os.path.join(BASE_DIR, "data", "DQN1_Dataset.xlsx")
SHEET_NAME = "Data"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Targets (as described by the project)
TARGETS_AIR = ["pm2.5", "no2", "co2"]
TARGET_HEALTH = "healthRiskScore"


def add_time_features(X: pd.DataFrame, dt: pd.Series) -> pd.DataFrame:
    """Adds simple calendar features derived from a datetime series."""
    X = X.copy()
    X["year"] = dt.dt.year
    X["dayofyear"] = dt.dt.dayofyear
    X["month"] = dt.dt.month
    X["day"] = dt.dt.day
    X["hour"] = dt.dt.hour
    return X


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error (works across scikit-learn versions)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape_percent(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error reported as percent."""
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)


def require_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------
    # 1) Load dataset
    # -----------------------------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            f"Put the Excel file here: {os.path.join(BASE_DIR, 'data')}\\DQN1_Dataset.xlsx"
        )

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    # We expect a timestamp column used earlier in the project: datetimeEpoch (seconds)
    require_columns(df, ["datetimeEpoch"], "time parsing")

    # Convert epoch seconds to datetime and sort for time-based split
    df["datetime"] = pd.to_datetime(df["datetimeEpoch"], unit="s", errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    # Ensure targets exist
    require_columns(df, TARGETS_AIR + [TARGET_HEALTH], "targets")

    # -----------------------------
    # 2) Build AIR feature matrix + target matrix
    # -----------------------------
    # Drop targets & non-feature time columns, then add derived time features
    X_air = df.drop(columns=TARGETS_AIR + [TARGET_HEALTH], errors="ignore")
    X_air = X_air.drop(columns=["datetimeEpoch", "datetime"], errors="ignore")
    X_air = add_time_features(X_air, df["datetime"])

    # Keep numeric features only (defensive against any string columns)
    X_air = X_air.select_dtypes(include=[np.number]).copy()
    Y_air = df[TARGETS_AIR].copy()

    # -----------------------------
    # 3) Build HEALTH feature matrix + target vector
    # -----------------------------
    X_health = df.drop(columns=[TARGET_HEALTH], errors="ignore")
    X_health = X_health.drop(columns=["datetimeEpoch", "datetime"], errors="ignore")
    X_health = add_time_features(X_health, df["datetime"])
    X_health = X_health.select_dtypes(include=[np.number]).copy()
    Y_health = df[TARGET_HEALTH].copy()

    # -----------------------------
    # 4) Time-based train/test split
    # -----------------------------
    split_idx = int(len(df) * 0.8)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Not enough rows after cleaning datetime to perform train/test split.")

    X_air_train, X_air_test = X_air.iloc[:split_idx], X_air.iloc[split_idx:]
    Y_air_train, Y_air_test = Y_air.iloc[:split_idx], Y_air.iloc[split_idx:]

    X_h_train, X_h_test = X_health.iloc[:split_idx], X_health.iloc[split_idx:]
    Y_h_train, Y_h_test = Y_health.iloc[:split_idx], Y_health.iloc[split_idx:]

    # -----------------------------
    # 5) Train Random Forest models
    # -----------------------------
    # AIR: multi-output RF
    air_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                MultiOutputRegressor(
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=42,
                        n_jobs=-1,
                    )
                ),
            ),
        ]
    )
    air_model.fit(X_air_train, Y_air_train)
    air_pred = air_model.predict(X_air_test)  # shape: (n_test, 3)

    # HEALTH: single-output RF
    health_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=400,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    health_model.fit(X_h_train, Y_h_train)
    health_pred = health_model.predict(X_h_test)  # shape: (n_test,)

    # -----------------------------
    # 6) Metrics (RMSE + MAPE%)
    # -----------------------------
    air_metrics_rows = []
    for i, target in enumerate(TARGETS_AIR):
        air_metrics_rows.append(
            {
                "target": target,
                "RMSE": rmse(Y_air_test[target].values, air_pred[:, i]),
                "MAPE_percent": mape_percent(Y_air_test[target].values, air_pred[:, i]),
            }
        )

    air_metrics_df = pd.DataFrame(air_metrics_rows)
    air_metrics_path = os.path.join(OUTPUT_DIR, "air_quality_metrics.csv")
    air_metrics_df.to_csv(air_metrics_path, index=False)

    health_metrics_df = pd.DataFrame(
        [
            {
                "target": TARGET_HEALTH,
                "RMSE": rmse(Y_h_test.values, health_pred),
                "MAPE_percent": mape_percent(Y_h_test.values, health_pred),
            }
        ]
    )
    health_metrics_path = os.path.join(OUTPUT_DIR, "health_risk_metrics.csv")
    health_metrics_df.to_csv(health_metrics_path, index=False)

    # -----------------------------
    # 7) Save predictions for trend analysis
    # -----------------------------
    dt_test = df["datetime"].iloc[split_idx:].values

    pred_air = pd.DataFrame(
        {
            "datetime": dt_test,
            "pm2.5_actual": Y_air_test["pm2.5"].values,
            "pm2.5_pred": air_pred[:, 0],
            "no2_actual": Y_air_test["no2"].values,
            "no2_pred": air_pred[:, 1],
            "co2_actual": Y_air_test["co2"].values,
            "co2_pred": air_pred[:, 2],
        }
    )
    pred_air_path = os.path.join(OUTPUT_DIR, "air_quality_predictions.csv")
    pred_air.to_csv(pred_air_path, index=False)

    pred_health = pd.DataFrame(
        {
            "datetime": dt_test,
            "healthRiskScore_actual": Y_h_test.values,
            "healthRiskScore_pred": health_pred,
        }
    )
    pred_health_path = os.path.join(OUTPUT_DIR, "health_risk_predictions.csv")
    pred_health.to_csv(pred_health_path, index=False)

    # -----------------------------
    # 8) Trend plots (actual vs predicted)
    # -----------------------------
    plot_specs = [
        ("pm2.5", "pm2.5_actual", "pm2.5_pred", "pm2.5"),
        ("no2", "no2_actual", "no2_pred", "no2"),
        ("co2", "co2_actual", "co2_pred", "co2"),
    ]

    for name, actual_col, pred_col, ylabel in plot_specs:
        plt.figure()
        plt.plot(pred_air["datetime"], pred_air[actual_col], label="Actual")
        plt.plot(pred_air["datetime"], pred_air[pred_col], label="Predicted")
        plt.title(f"Actual vs Predicted {name} (Test Set)")
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"trend_{name}.png"), dpi=160)
        plt.close()

    plt.figure()
    plt.plot(pred_health["datetime"], pred_health["healthRiskScore_actual"], label="Actual")
    plt.plot(pred_health["datetime"], pred_health["healthRiskScore_pred"], label="Predicted")
    plt.title("Actual vs Predicted Health Risk Score (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("healthRiskScore")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "trend_healthRiskScore.png"), dpi=160)
    plt.close()

    # Final console message
    print("Success: check outputs/ for metrics, prediction files, and plots.")
    print(f"- {air_metrics_path}")
    print(f"- {health_metrics_path}")
    print(f"- {pred_air_path}")
    print(f"- {pred_health_path}")


if __name__ == "__main__":
    main()
