import re
import ast
import pandas as pd


def analyze_hyperparam_log(log_path: str):
    # Compile regular expressions for efficient pattern matching
    # Matches the configuration dictionary string
    config_pattern = re.compile(r"\[\d+/\d+\] Config:\s*(\{.*\})")
    # Matches the validation loss float value
    loss_pattern = re.compile(r"-> (?:Average|Best) Val Loss:\s*([0-9.]+)")

    results = []
    current_config = None

    # Parse the log file line by line to maintain low memory footprint
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Attempt to extract the hyperparameter dictionary
            config_match = config_pattern.search(line)
            if config_match:
                try:
                    # Safely evaluate the string to a Python dictionary object
                    current_config = ast.literal_eval(config_match.group(1))
                except (ValueError, SyntaxError) as e:
                    print(
                        f"Failed to parse config: {config_match.group(1)} | Error: {e}"
                    )
                    current_config = None
                continue

            # Attempt to extract the objective function value (Validation Loss)
            loss_match = loss_pattern.search(line)
            if loss_match and current_config is not None:
                val_loss = float(loss_match.group(1))

                # Construct an observation combining hyperparameters and their resultant loss
                entry = current_config.copy()
                entry["best_val_loss"] = val_loss
                results.append(entry)

                # Reset the state to prevent misalignment of mismatched log entries
                current_config = None

    if not results:
        print("No valid configuration data found in the provided log file.")
        return None

    # Convert the list of dictionaries into a pandas DataFrame for vectorized statistical operations
    df = pd.DataFrame(results)

    # Compute empirical mean and sample variance across the entire hyperparameter search space
    overall_mean = df["best_val_loss"].mean()
    overall_variance = df["best_val_loss"].var()

    print("\n=== Global Loss Landscape Analysis ===")
    print(f"Total configurations parsed : {len(df)}")
    print(f"Expected Val Loss (Mean)    : {overall_mean:.6f}")
    print(f"Dispersion (Variance)       : {overall_variance:.6e}\n")

    # Generate all combinations of hyperparameters (size 1 to N) to find the best subsets
    critical_params = [col for col in df.columns if col != "best_val_loss"]
    
    import itertools
    all_results = []
    
    # To prevent exponential explosion if there are too many hyperparameters
    max_k = min(len(critical_params), 8) 
    
    for k in range(1, max_k + 1):
        for subset in itertools.combinations(critical_params, k):
            subset = list(subset)
            grouped = df.groupby(subset)["best_val_loss"].agg(
                Mean="mean",
                Std="std",
                Count="count"
            ).reset_index()
            
            for _, row in grouped.iterrows():
                config_str = ", ".join([f"{p}={row[p]}" for p in subset])
                all_results.append({
                    "Num_Params": k,
                    "Config": config_str,
                    "Mean": row["Mean"],
                    "Std": row["Std"],
                    "Count": row["Count"]
                })
                
    res_df = pd.DataFrame(all_results)
    
    import math
    print("\n=== Top 10 Hyperparameter Combinations by Mean Loss ===")
    top_mean = res_df.sort_values(by="Mean", ascending=True).head(10)
    for i, row in enumerate(top_mean.itertuples(), 1):
        # If count is 1, std is NaN, so we display 'inf' as requested
        std_str = "inf" if row.Count == 1 else f"{row.Std:.6e}"
        print(f"Rank {i:2d} | Mean Loss: {row.Mean:.6f} | Std Dev: {std_str:>12} | Count: {int(row.Count):2d}")
        print(f"        | Config: {row.Config}\n")

    return df


if __name__ == "__main__":
    import argparse
    
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
