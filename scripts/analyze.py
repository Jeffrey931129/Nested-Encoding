import argparse
import ast
import itertools
import re

import pandas as pd


def analyze_hyperparam_log(log_path: str):
    config_pattern = re.compile(r"\[\d+/\d+\] Config:\s*(\{.*\})")
    loss_pattern = re.compile(r"Test Loss:\s*([0-9.]+)")
    results = []
    current_config = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            config_match = config_pattern.search(line)
            if config_match:
                try:
                    current_config = ast.literal_eval(config_match.group(1))
                except (ValueError, SyntaxError) as e:
                    print(f"Failed to parse config: {config_match.group(1)} | Error: {e}")
                    current_config = None
                continue
            loss_match = loss_pattern.search(line)
            if loss_match and current_config is not None:
                test_loss = float(loss_match.group(1))
                entry = current_config.copy()
                entry["test_loss"] = test_loss
                results.append(entry)
                current_config = None

    if not results:
        print("No valid configuration data found in the provided log file.")
        return None

    df = pd.DataFrame(results)

    overall_mean = df["test_loss"].mean()
    overall_variance = df["test_loss"].var()

    print("\n=== Global Loss Landscape Analysis ===")
    print(f"Total configurations parsed : {len(df)}")
    print(f"Expected Test Loss (Mean)   : {overall_mean:.6f}")
    print(f"Dispersion (Variance)       : {overall_variance:.6e}\n")

    critical_params = [col for col in df.columns if col != "test_loss" and df[col].astype(str).nunique() > 1]

    all_results = []

    max_k = min(len(critical_params), 8)

    for k in range(1, max_k + 1):
        for subset in itertools.combinations(critical_params, k):
            subset = list(subset)
            grouped = df.groupby(subset)["test_loss"].agg(Mean="mean", Std="std", Count="count").reset_index()

            for _, row in grouped.iterrows():
                config_str = ", ".join([f"{p}={row[p]}" for p in subset])
                all_results.append({"Num_Params": k, "Config": config_str, "Mean": row["Mean"], "Std": row["Std"], "Count": row["Count"]})

    res_df = pd.DataFrame(all_results)

    print("\n=== Top 10 Hyperparameter Combinations by Mean Loss ===")
    top_mean = res_df.sort_values(by="Mean", ascending=True).head(10)
    for i, row in enumerate(top_mean.itertuples(), 1):
        std_str = "inf" if row.Count == 1 else f"{row.Std:.6e}"
        print(f"Rank {i:2d} | Mean Loss: {row.Mean:.6f} | Std Dev: {std_str:>12} | Count: {int(row.Count):2d}")
        print(f"        | Config: {row.Config}\n")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze hyperparameter log file.")
    parser.add_argument(
        "log_path",
        type=str,
        nargs="?",
        help="Path to the log file to analyze",
    )
    args = parser.parse_args()

    if not args.log_path:
        args.log_path = input("Please enter log file path: ").strip()

    if args.log_path:
        df_results = analyze_hyperparam_log(args.log_path)
    else:
        print("No path provided, exiting.")
