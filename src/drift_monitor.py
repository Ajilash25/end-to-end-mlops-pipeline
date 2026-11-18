
import os
import json
import pandas as pd

from scipy.stats import ks_2samp
from evidently import Report
from evidently.presets import DataDriftPreset


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(REPORT_DIR, exist_ok=True)

REFERENCE_PATH = os.path.join(
    DATA_DIR, "reference_data.csv"
)

CURRENT_PATH = os.path.join(
    DATA_DIR, "current_data.csv"
)

HTML_REPORT_PATH = os.path.join(
    REPORT_DIR, "drift_report.html"
)

JSON_REPORT_PATH = os.path.join(
    REPORT_DIR, "drift_result.json"
)


# ==========================================
# MAIN DRIFT MONITORING
# ==========================================

def main():

    print("\n========== DATA DRIFT MONITORING ==========")

    # Load datasets
    reference_data = pd.read_csv(REFERENCE_PATH)
    current_data = pd.read_csv(CURRENT_PATH)

    print("\nReference data:", reference_data.shape)
    print("Current data:", current_data.shape)

    # Remove target column from current dataset
    if "target" in current_data.columns:
        current_data = current_data.drop(columns=["target"])
        print("Target column excluded from drift monitoring.")

    # Validate feature columns
    if list(reference_data.columns) != list(current_data.columns):
        raise ValueError("Reference and current feature columns do not match.")

    print("\nGenerating Evidently drift report...")

    report = Report([
        DataDriftPreset()
    ])

    report_result = report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    report_result.save_html(HTML_REPORT_PATH)

    print("Evidently report generated!")
    print("HTML report:", HTML_REPORT_PATH)

    # ======================================
    # STATISTICAL DRIFT CHECK
    # ======================================

    print("\n========== DRIFT RESULTS ==========")

    drifted_features = []

    for column in reference_data.columns:

        statistic, p_value = ks_2samp(
            reference_data[column],
            current_data[column]
        )

        if p_value < 0.05:
            drifted_features.append(column)

            print(
                f"{column}: DRIFT DETECTED "
                f"(p-value: {p_value:.6f})"
            )

        else:
            print(
                f"{column}: STABLE "
                f"(p-value: {p_value:.6f})"
            )

    total_features = len(reference_data.columns)
    drifted_count = len(drifted_features)

    drift_share = (
        drifted_count / total_features
        if total_features > 0 else 0
    )

    # Dataset drift threshold: 50%
    drift_detected = drift_share >= 0.5

    print("\nDrifted features:", drifted_count)
    print("Total features:", total_features)
    print(f"Drift share: {drift_share:.2%}")
    print("Dataset drift:", drift_detected)

    # ======================================
    # SAVE JSON RESULT
    # ======================================

    drift_result = {
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "drifted_features": drifted_features,
        "total_features": total_features,
        "drifted_count": drifted_count
    }

    with open(
        JSON_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            drift_result,
            file,
            indent=4
        )

    print("\nDrift result saved:", JSON_REPORT_PATH)

    # ======================================
    # GITHUB ACTIONS OUTPUT
    # ======================================

    github_output = os.getenv("GITHUB_OUTPUT")

    if github_output:

        with open(
            github_output,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                "drift_detected="
                + str(drift_detected).lower()
                + "\n"
            )

        print("GitHub Actions output updated.")

    print("\nDRIFT MONITORING COMPLETED")


if __name__ == "__main__":
    main()
