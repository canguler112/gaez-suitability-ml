# test_rf_xgb_significance.py
#
# Purpose:
# Fold-level paired statistical comparison between RF-rs03 and XGB-rs04.
# The RF prediction file is stored with the rsbest filename for compatibility.
# and XGB-rs04 under 2.0 degree spatial block cross-validation.
#
# Tests:
# - Paired t-test
# - Wilcoxon signed-rank test
#
# Metrics:
# - R2
# - RMSE
# - MAE
#
# Effect size:
# - Mean fold-level difference
# - Cohen's dz for paired differences
#
# Interpretation:
# For RMSE and MAE:
#   positive difference = RF error - XGB error = XGB improvement
#
# For R2:
#   positive difference = XGB R2 - RF R2 = XGB improvement

import os
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_DIR = r"C:\Users\cangu\OneDrive\Desktop\Agriculture"

RF_PRED_PATH = os.path.join(
    PROJECT_DIR,
    r"outputs\analysis\rf_rsbest_block2p0_cv_predictions.csv"
)

XGB_PRED_PATH = os.path.join(
    PROJECT_DIR,
    r"outputs\analysis\xgb_rs04_block2p0_cv_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    r"outputs\analysis\significance_rf_xgb"
)


# =========================================================
# HELPERS
# =========================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def compute_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def find_column(df, candidates, required=True):
    """
    Find a likely column name from a list of candidates.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def load_prediction_file(path, model_name):
    print(f"\nLoading {model_name} predictions:")
    print(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)
    print(f"{model_name} shape:", df.shape)
    print(f"{model_name} columns:", list(df.columns))

    fold_col = find_column(df, ["fold", "cv_fold", "Fold"])
    obs_col = find_column(df, ["observed", "y_true", "true", "actual", "suitability"])
    pred_col = find_column(df, ["predicted", "y_pred", "prediction", "pred"])

    out = df[[fold_col, obs_col, pred_col]].copy()
    out.columns = ["fold", "observed", "predicted"]

    out["fold"] = out["fold"].astype(int)
    out["observed"] = out["observed"].astype(float)
    out["predicted"] = out["predicted"].astype(float)

    print(f"{model_name} folds:", sorted(out["fold"].unique()))

    return out


def compute_fold_metrics(pred_df, model_name):
    rows = []

    for fold, g in pred_df.groupby("fold"):
        m = compute_metrics(g["observed"], g["predicted"])
        m["fold"] = int(fold)
        m["model"] = model_name
        m["n_test"] = int(len(g))
        rows.append(m)

    df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)

    return df[["model", "fold", "n_test", "r2", "rmse", "mae"]]


def cohen_dz(diff):
    """
    Cohen's dz for paired samples:
        mean(diff) / sd(diff)
    """
    diff = np.asarray(diff, dtype=float)

    if len(diff) < 2:
        return np.nan

    sd = np.std(diff, ddof=1)

    if sd == 0:
        return np.nan

    return float(np.mean(diff) / sd)


def run_paired_tests(diff, alternative="greater"):
    """
    Run paired tests on already-computed improvement differences.

    For this thesis:
    - RMSE diff = RF RMSE - XGB RMSE
    - MAE diff = RF MAE - XGB MAE
    - R2 diff = XGB R2 - RF R2

    Therefore positive diff means XGB improved over RF.

    We report:
    - two-sided paired t-test
    - one-sided paired t-test in expected direction
    - two-sided Wilcoxon
    - one-sided Wilcoxon in expected direction
    """
    diff = np.asarray(diff, dtype=float)

    n = len(diff)
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else np.nan
    dz = cohen_dz(diff)

    # Paired t-test on differences against zero.
    # scipy ttest_1samp is equivalent to paired t-test on pairwise differences.
    t_two = stats.ttest_1samp(diff, popmean=0.0, alternative="two-sided")
    t_one = stats.ttest_1samp(diff, popmean=0.0, alternative=alternative)

    # Wilcoxon signed-rank test.
    # With n=5, p-values should be interpreted cautiously.
    try:
        w_two = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", mode="auto")
        w_one = stats.wilcoxon(diff, zero_method="wilcox", alternative=alternative, mode="auto")
        wilcoxon_stat_two = float(w_two.statistic)
        wilcoxon_p_two = float(w_two.pvalue)
        wilcoxon_stat_one = float(w_one.statistic)
        wilcoxon_p_one = float(w_one.pvalue)
    except ValueError as e:
        wilcoxon_stat_two = np.nan
        wilcoxon_p_two = np.nan
        wilcoxon_stat_one = np.nan
        wilcoxon_p_one = np.nan
        print("Wilcoxon warning:", e)

    return {
        "n_folds": int(n),
        "mean_difference": mean_diff,
        "sd_difference": sd_diff,
        "cohen_dz": dz,
        "paired_t_stat_two_sided": float(t_two.statistic),
        "paired_t_p_two_sided": float(t_two.pvalue),
        "paired_t_stat_one_sided_xgb_better": float(t_one.statistic),
        "paired_t_p_one_sided_xgb_better": float(t_one.pvalue),
        "wilcoxon_stat_two_sided": wilcoxon_stat_two,
        "wilcoxon_p_two_sided": wilcoxon_p_two,
        "wilcoxon_stat_one_sided_xgb_better": wilcoxon_stat_one,
        "wilcoxon_p_one_sided_xgb_better": wilcoxon_p_one,
        "all_differences_positive": bool(np.all(diff > 0)),
    }


def make_comparison_table(rf_metrics, xgb_metrics):
    merged = rf_metrics.merge(
        xgb_metrics,
        on="fold",
        suffixes=("_rf", "_xgb")
    )

    if len(merged) == 0:
        raise ValueError("No matching folds found between RF and XGB metrics.")

    # XGB improvement definitions
    merged["delta_rmse_rf_minus_xgb"] = merged["rmse_rf"] - merged["rmse_xgb"]
    merged["delta_mae_rf_minus_xgb"] = merged["mae_rf"] - merged["mae_xgb"]
    merged["delta_r2_xgb_minus_rf"] = merged["r2_xgb"] - merged["r2_rf"]

    return merged


def write_interpretation_txt(results_df, comparison_df, output_path):
    lines = []

    lines.append("RF-rs03 / RF-rsbest vs XGB-rs04 significance check")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Validation setup: 2.0 degree spatial block cross-validation")
    lines.append("Comparison type: fold-level paired comparison")
    lines.append("")
    lines.append("Important interpretation note:")
    lines.append(
        "Only five spatial folds are available, so p-values have low statistical power. "
        "The mean fold-level differences should be treated as the main effect-size evidence, "
        "with p-values used only as supporting evidence."
    )
    lines.append("")

    lines.append("Fold-level differences:")
    lines.append(
        "For RMSE and MAE, positive values mean RF error minus XGB error, so positive = XGB lower error."
    )
    lines.append(
        "For R2, positive values mean XGB R2 minus RF R2, so positive = XGB higher R2."
    )
    lines.append("")

    for _, row in results_df.iterrows():
        metric = row["metric"]
        mean_diff = row["mean_difference"]
        p_t_two = row["paired_t_p_two_sided"]
        p_w_two = row["wilcoxon_p_two_sided"]
        dz = row["cohen_dz"]

        lines.append(f"{metric}:")
        lines.append(f"  Mean difference: {mean_diff:.6f}")
        lines.append(f"  Cohen dz: {dz:.6f}")
        lines.append(f"  Paired t-test p-value two-sided: {p_t_two:.6f}")
        lines.append(f"  Wilcoxon p-value two-sided: {p_w_two:.6f}")
        lines.append(f"  All fold differences positive: {row['all_differences_positive']}")
        lines.append("")

    # Short interpretation
    lines.append("Interpretation:")
    lines.append(
        "XGB-rs04 was compared with RF-rs03/RF-rsbest using fold-level paired tests "
        "under the same 2.0 degree spatial block cross-validation folds. "
        "The main evidence should be the mean fold-level improvement in RMSE, MAE, and R2. "
        "Because n=5 folds is small, statistical significance should be interpreted cautiously."
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("SCRIPT STARTED: RF vs XGB fold-level significance tests")
    print("PROJECT_DIR:", PROJECT_DIR)
    print("RF_PRED_PATH:", RF_PRED_PATH)
    print("XGB_PRED_PATH:", XGB_PRED_PATH)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    ensure_dir(OUTPUT_DIR)

    # Load predictions
    rf_preds = load_prediction_file(RF_PRED_PATH, "RF-rs03/RF-rsbest")
    xgb_preds = load_prediction_file(XGB_PRED_PATH, "XGB-rs04")

    # Compute fold-level metrics
    rf_metrics = compute_fold_metrics(rf_preds, "RF-rs03/RF-rsbest")
    xgb_metrics = compute_fold_metrics(xgb_preds, "XGB-rs04")

    print("\nRF fold metrics:")
    print(rf_metrics.to_string(index=False))

    print("\nXGB fold metrics:")
    print(xgb_metrics.to_string(index=False))

    # Save fold-level metrics
    rf_metrics_path = os.path.join(OUTPUT_DIR, "rf_fold_metrics.csv")
    xgb_metrics_path = os.path.join(OUTPUT_DIR, "xgb_fold_metrics.csv")

    rf_metrics.to_csv(rf_metrics_path, index=False)
    xgb_metrics.to_csv(xgb_metrics_path, index=False)

    # Pair folds and compute differences
    comparison_df = make_comparison_table(rf_metrics, xgb_metrics)

    comparison_path = os.path.join(OUTPUT_DIR, "rf_xgb_fold_metric_differences.csv")
    comparison_df.to_csv(comparison_path, index=False)

    print("\nFold-level paired comparison:")
    print(comparison_df.to_string(index=False))

    # Run tests
    test_rows = []

    metrics_to_test = {
        "RMSE_RF_minus_XGB": comparison_df["delta_rmse_rf_minus_xgb"].values,
        "MAE_RF_minus_XGB": comparison_df["delta_mae_rf_minus_xgb"].values,
        "R2_XGB_minus_RF": comparison_df["delta_r2_xgb_minus_rf"].values,
    }

    for metric_name, diff in metrics_to_test.items():
        res = run_paired_tests(diff, alternative="greater")
        res["metric"] = metric_name
        test_rows.append(res)

    results_df = pd.DataFrame(test_rows)

    # Reorder columns
    cols = [
        "metric",
        "n_folds",
        "mean_difference",
        "sd_difference",
        "cohen_dz",
        "paired_t_stat_two_sided",
        "paired_t_p_two_sided",
        "paired_t_stat_one_sided_xgb_better",
        "paired_t_p_one_sided_xgb_better",
        "wilcoxon_stat_two_sided",
        "wilcoxon_p_two_sided",
        "wilcoxon_stat_one_sided_xgb_better",
        "wilcoxon_p_one_sided_xgb_better",
        "all_differences_positive",
    ]

    results_df = results_df[cols]

    results_path = os.path.join(OUTPUT_DIR, "rf_xgb_paired_tests.csv")
    results_df.to_csv(results_path, index=False)

    print("\nPaired test results:")
    print(results_df.to_string(index=False))

    # Save JSON too
    json_path = os.path.join(OUTPUT_DIR, "rf_xgb_paired_tests.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=2)

    # Interpretation notes
    notes_path = os.path.join(OUTPUT_DIR, "interpretation_notes.txt")
    write_interpretation_txt(results_df, comparison_df, notes_path)

    print("\nSaved outputs:")
    print("RF fold metrics:", rf_metrics_path)
    print("XGB fold metrics:", xgb_metrics_path)
    print("Fold differences:", comparison_path)
    print("Paired tests CSV:", results_path)
    print("Paired tests JSON:", json_path)
    print("Interpretation notes:", notes_path)

    print("\nDONE")


if __name__ == "__main__":
    main()
