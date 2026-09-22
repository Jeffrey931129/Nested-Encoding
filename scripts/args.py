import argparse

import numpy as np


def print_npy_args(npy_path: str):
    try:
        data = np.load(npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Failed to load {npy_path}: {e}")
        return

    if "args" not in data:
        print(f"No 'args' key found in {npy_path}.")
        return

    args_dict = data["args"]
    
    print(f"\n=== Arguments in {npy_path} ===")
    for key, val in args_dict.items():
        print("{:20} {}".format(key, val))
    print("================================{}\n".format("=" * len(npy_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read .npy file and print args.")
    parser.add_argument(
        "npy_path",
        type=str,
        nargs="?",
        help="Path to the .npy file to read",
    )
    args = parser.parse_args()

    if not args.npy_path:
        args.npy_path = input("Please enter .npy file path: ").strip()

    if args.npy_path:
        print_npy_args(args.npy_path)
    else:
        print("No path provided, exiting.")

