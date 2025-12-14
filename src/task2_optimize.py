import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "task2")

AIR_METRICS = os.path.join(OUT_DIR, "task2_air_metrics.csv")
HEALTH_METRICS = os.path.join(OUT_DIR, "task2_health_metrics.csv")


def main():
    if not os.path.exists(AIR_METRICS) or not os.path.exists(HEALTH_METRICS):
        raise FileNotFoundError("Run the C1 version first to generate task2_air_metrics.csv and task2_health_metrics.csv")

    air = pd.read_csv(AIR_METRICS)
    health = pd.read_csv(HEALTH_METRICS)

    # Baseline row
    baseline = health[health["variant"] == "baseline"].iloc[0]

    # Best optimized = lowest health RMSE (excluding baseline)
    candidates = health[health["variant"] != "baseline"].copy()
    best = candidates.sort_values("RMSE").iloc[0]

    comparison = {
        "baseline_variant": "baseline",
        "best_variant": best["variant"],
        "baseline_health_RMSE": float(baseline["RMSE"]),
        "best_health_RMSE": float(best["RMSE"]),
        "baseline_health_MAPE_percent": float(baseline["MAPE_percent"]),
        "best_health_MAPE_percent": float(best["MAPE_percent"]),
        "health_RMSE_improvement": float(baseline["RMSE"] - best["RMSE"]),
        "health_RMSE_improvement_percent": float((baseline["RMSE"] - best["RMSE"]) / baseline["RMSE"] * 100.0),
        "health_MAPE_improvement_percent": float((baseline["MAPE_percent"] - best["MAPE_percent"]) / baseline["MAPE_percent"] * 100.0),
    }

    # Add air target comparisons for the same "best_variant" where available
    for tgt in air["target"].unique():
        b_row = air[(air["variant"] == "baseline") & (air["target"] == tgt)].iloc[0]
        v_row = air[(air["variant"] == best["variant"]) & (air["target"] == tgt)]
        if len(v_row) == 1:
            v_row = v_row.iloc[0]
            comparison[f"{tgt}_baseline_RMSE"] = float(b_row["RMSE"])
            comparison[f"{tgt}_best_RMSE"] = float(v_row["RMSE"])
            comparison[f"{tgt}_RMSE_improvement_percent"] = float((b_row["RMSE"] - v_row["RMSE"]) / b_row["RMSE"] * 100.0)

    out_path = os.path.join(OUT_DIR, "task2_comparison_before_after.csv")
    pd.DataFrame([comparison]).to_csv(out_path, index=False)

    print("C2 complete: wrote before/after comparison to outputs/task2/task2_comparison_before_after.csv")
    print(f"Best variant selected by lowest health RMSE: {best['variant']}")


if __name__ == "__main__":
    main()
