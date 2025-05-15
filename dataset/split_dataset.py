import os
import random
import json
import argparse

def split_files(file_list, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    """
    Splits a list of files into training, development, and test sets based on the given ratios.

    :param file_list: List of file paths.
    :param train_ratio: Ratio of files to be used for training (default is 0.8).
    :param dev_ratio: Ratio of files to be used for development (default is 0.1).
    :param test_ratio: Ratio of files to be used for testing (default is 0.1).
    :return: Dictionary containing the split file names.
    """
    # Ensure the ratios sum to 1
    assert train_ratio + dev_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Shuffle the list of files to ensure randomness
    random.shuffle(file_list)

    # Calculate the number of files for each set
    total_files = len(file_list)
    train_size = int(total_files * train_ratio)
    dev_size = int(total_files * dev_ratio)

    # Split the files into training, development, and test sets
    train_files = [os.path.basename(fp) for fp in file_list[:train_size]]
    dev_files = [os.path.basename(fp) for fp in file_list[train_size:train_size + dev_size]]
    test_files = [os.path.basename(fp) for fp in file_list[train_size + dev_size:]]

    # Create a dictionary to store the split file names
    split_dict = {
        "train": train_files,
        "dev": dev_files,
        "test": test_files
    }

    return split_dict

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Split files into training, development, and test sets.")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing the files to split.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the split information JSON file.")
    args = parser.parse_args()

    # Get the list of files in the directory
    file_list = [os.path.join(args.directory, f) for f in os.listdir(args.directory) if os.path.isfile(os.path.join(args.directory, f))]

    # Split the files into training, dev, and test sets
    split_dict = split_files(file_list)

    # Save the dictionary to a JSON file
    with open(args.output, "w") as f:
        json.dump(split_dict, f, indent=4)

    print(f"File splits saved to {args.output}")

    # Print the dictionary for verification
    print("Training files:", len(split_dict["train"]))
    print("Development files:", len(split_dict["dev"]))
    print("Test files:", len(split_dict["test"]))

if __name__ == "__main__":
    main()