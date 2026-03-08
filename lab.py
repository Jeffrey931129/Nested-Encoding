import subprocess

dnn = "alexnet+nested"

# Step 1: Synthesizing EEG data
command_1 = [
    "python", "04_synthesizing_eeg_data/end_to_end_encoding.py",
    "--dnn", dnn
]
print(f"Running: {' '.join(command_1)}")
subprocess.run(command_1, check=True)

# Step 2: Correlation Analysis
command_2 = [
    "python", "05_synthetic_data_analyses/correlation.py",
    "--dnn", dnn
]
print(f"Running: {' '.join(command_2)}")
subprocess.run(command_2, check=True)

# Step 3: Correlation Stats Analysis
command_3 = [
    "python", "05_synthetic_data_analyses/correlation_stats.py",
    "--dnn", dnn
]
print(f"Running: {' '.join(command_3)}")
subprocess.run(command_3, check=True)

# Step 4: Plotting
command_4 = [
    "python", "06_plotting/plot.py"
]
print(f"Running: {' '.join(command_4)}")
subprocess.run(command_4, check=True)