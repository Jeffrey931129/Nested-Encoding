import re
import ast
import pandas as pd


def analyze_hyperparam_log(log_path: str):
    # Compile regular expressions for efficient pattern matching
    # Matches the configuration dictionary string
    config_pattern = re.compile(r"\[\d+/\d+\] Config:\s*(\{.*\})")
    # Matches the validation loss float value
    loss_pattern = re.compile(r"-> Best Val Loss:\s*([0-9.]+)")

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

    print("=== Global Loss Landscape Analysis ===")
    print(f"Total configurations parsed : {len(df)}")
    print(f"Expected Val Loss (Mean)    : {overall_mean:.6f}")
    print(f"Dispersion (Variance)       : {overall_variance:.6e}\n")

    # Perform marginal statistical analysis on critical hyperparameter subsets
    # This isolates the effect of individual hyperparameters (e.g., learning rate)
    # by computing conditional expectation and conditional variance.
    critical_params = [col for col in df.columns if col != "best_val_loss"]

    for param in critical_params:
        if param in df.columns:
            print(f"=== Marginal Analysis conditional on '{param}' ===")
            # Aggregate metrics to evaluate hyperparameter sensitivity and robustness
            stats_df = (
                df.groupby(param)["best_val_loss"]
                .agg(
                    Conditional_Mean="mean",
                    Conditional_Variance="var",
                    Sample_Size="count",
                )
                .reset_index()
            )
            print(
                stats_df.to_string(
                    index=False,
                    justify="right",
                    formatters={
                        "Conditional_Mean": "{:.6f}".format,
                        "Conditional_Variance": "{:.6e}".format,
                    },
                )
            )
            print("\n")

    return df


# Example Execution:
# Assuming your log file is named 'hparam_search.log'
if __name__ == "__main__":
    # Uncomment the line below to run with your specific log file path
    df_results = analyze_hyperparam_log(r"experiment\nested_adam + nested_model\5conv2d + 3LSTM (10M)\.log")
    pass
