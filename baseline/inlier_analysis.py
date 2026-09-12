"""Analyze R@1 correctness vs. number of inliers."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", required=True)
    parser.add_argument("--inliers-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="inlier_analysis_outputs",
    )
    return parser.parse_args()


def parse_prediction_file(txt_file):
    """Read predictions and positives from a .txt file."""

    lines = txt_file.read_text(encoding="utf-8").splitlines()

    predictions = []
    positives = []

    section = None

    for line in lines:
        line = line.strip()

        if line == "Predictions paths:":
            section = "predictions"
            continue

        if line == "Positives paths:":
            section = "positives"
            continue

        if not line:
            continue

        if section == "predictions":
            predictions.append(Path(line).name)

        elif section == "positives":
            positives.append(Path(line).name)

    return predictions, positives


def is_r1_correct(predictions, positives):
    """Check whether the first retrieved image is a positive."""

    if not predictions or not positives:
        return False

    return predictions[0] in set(positives)


def load_inliers(inliers_dir):
    """Load first-result inlier count from each matching .torch file."""

    inliers_dir = Path(inliers_dir)

    values = {}

    for file_path in sorted(inliers_dir.glob("*.torch")):
        try:
            results = torch.load(file_path, weights_only=False)

            if len(results) == 0:
                num_inliers = 0
            else:
                num_inliers = results[0]["num_inliers"]

            query_id = file_path.stem
            values[query_id] = int(num_inliers)

        except Exception as e:
            print(f"Warning: failed to load {file_path}: {e}")

    return values


def analyze(preds_dir, inliers_dir):
    """Match prediction correctness with inlier counts."""

    preds_dir = Path(preds_dir)

    inliers = load_inliers(inliers_dir)

    correct = []
    wrong = []

    txt_files = sorted(preds_dir.glob("*.txt"))

    for txt_file in txt_files:
        query_id = txt_file.stem

        if query_id not in inliers:
            continue

        predictions, positives = parse_prediction_file(txt_file)

        if is_r1_correct(predictions, positives):
            correct.append(inliers[query_id])
        else:
            wrong.append(inliers[query_id])

    return np.asarray(correct), np.asarray(wrong)


def print_statistics(correct, wrong):
    print("\n========== Inlier Statistics ==========")

    print(f"Correct queries: {len(correct)}")
    print(f"Wrong queries:   {len(wrong)}")

    if len(correct) > 0:
        print(f"Correct mean:   {np.mean(correct):.2f}")
        print(f"Correct median: {np.median(correct):.2f}")

    if len(wrong) > 0:
        print(f"Wrong mean:     {np.mean(wrong):.2f}")
        print(f"Wrong median:   {np.median(wrong):.2f}")


def plot_histogram(correct, wrong, output_dir):
    """Create histogram comparing correct and wrong R@1."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_values = np.concatenate([correct, wrong])

    if len(all_values) == 0:
        print("No data available for histogram.")
        return

    bins = np.histogram_bin_edges(all_values, bins=30)

    plt.figure(figsize=(8, 5))

    plt.hist(
        correct,
        bins=bins,
        alpha=0.6,
        label="Correct R@1",
    )

    plt.hist(
        wrong,
        bins=bins,
        alpha=0.6,
        label="Wrong R@1",
    )

    plt.xlabel("Number of Inliers")
    plt.ylabel("Number of Queries")
    plt.title("R@1 Correctness vs. Inlier Count")
    plt.legend()

    plt.tight_layout()

    output_file = output_dir / "inlier_histogram.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"\nHistogram saved to: {output_file}")


def main():
    args = parse_args()

    correct, wrong = analyze(
        args.preds_dir,
        args.inliers_dir,
    )

    print_statistics(correct, wrong)

    plot_histogram(
        correct,
        wrong,
        args.output_dir,
    )


if __name__ == "__main__":
    main()