"""Find an inlier threshold that separates correct and wrong R@1 retrievals."""

import argparse
from pathlib import Path
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find the best inlier threshold for R@1 correctness."
    )
    parser.add_argument("--preds-dir", required=True)
    parser.add_argument("--inliers-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="threshold_analysis_outputs"
    )
    return parser.parse_args()


def normalize_path(path):
    return Path(path.strip()).name


def load_data(preds_dir, inliers_dir):
    preds_dir = Path(preds_dir)
    inliers_dir = Path(inliers_dir)

    pred_files = sorted(preds_dir.glob("*.txt"))
    inlier_files = sorted(inliers_dir.glob("*.torch"))

    n = min(len(pred_files), len(inlier_files))

    inliers = []
    correct = []

    for pred_file, inlier_file in zip(pred_files[:n], inlier_files[:n]):

        # ---------- Read prediction file ----------
        text = pred_file.read_text(errors="ignore")
        lines = text.splitlines()

        prediction_paths = []
        positive_paths = []

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
                prediction_paths.append(normalize_path(line))

            elif section == "positives":
                positive_paths.append(normalize_path(line))

        if not prediction_paths or not positive_paths:
            continue

        # R@1 is correct if first prediction is one of positives
        rank1 = prediction_paths[0]
        is_correct = rank1 in positive_paths

        # ---------- Read matching result ----------
        try:
            results = torch.load(
                inlier_file,
                weights_only=False
            )

            if len(results) == 0:
                num_inliers = 0
            else:
                num_inliers = results[0]["num_inliers"]

        except Exception:
            continue

        inliers.append(float(num_inliers))
        correct.append(int(is_correct))

    return np.asarray(inliers), np.asarray(correct)


def find_best_threshold(inliers, correct):

    if len(inliers) == 0:
        return None

    best = None

    min_value = int(np.min(inliers))
    max_value = int(np.max(inliers))

    for threshold in range(min_value, max_value + 1):

        predicted_correct = inliers >= threshold

        tp = np.sum(predicted_correct & (correct == 1))
        tn = np.sum((~predicted_correct) & (correct == 0))
        fp = np.sum(predicted_correct & (correct == 0))
        fn = np.sum((~predicted_correct) & (correct == 1))

        accuracy = (tp + tn) / len(correct)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        result = {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

        if best is None or result["f1"] > best["f1"]:
            best = result

    return best


def main():

    args = parse_args()

    inliers, correct = load_data(
        args.preds_dir,
        args.inliers_dir
    )

    print("\n========== Threshold Analysis ==========")
    print(f"Total queries: {len(correct)}")
    print(f"Correct queries: {np.sum(correct == 1)}")
    print(f"Wrong queries:   {np.sum(correct == 0)}")

    if len(correct) == 0:
        print("No data available.")
        return

    best = find_best_threshold(inliers, correct)

    print("\n---------- Best Threshold ----------")
    print(f"Threshold: {best['threshold']}")
    print(f"Accuracy:  {best['accuracy']:.4f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall:    {best['recall']:.4f}")
    print(f"F1-score:  {best['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(f"TP: {best['tp']}")
    print(f"TN: {best['tn']}")
    print(f"FP: {best['fp']}")
    print(f"FN: {best['fn']}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "best_threshold.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Threshold Analysis\n")
        f.write("==================\n")
        f.write(f"Total queries: {len(correct)}\n")
        f.write(f"Correct queries: {np.sum(correct == 1)}\n")
        f.write(f"Wrong queries: {np.sum(correct == 0)}\n\n")

        f.write(f"Best threshold: {best['threshold']}\n")
        f.write(f"Accuracy: {best['accuracy']:.4f}\n")
        f.write(f"Precision: {best['precision']:.4f}\n")
        f.write(f"Recall: {best['recall']:.4f}\n")
        f.write(f"F1-score: {best['f1']:.4f}\n\n")

        f.write(f"TP: {best['tp']}\n")
        f.write(f"TN: {best['tn']}\n")
        f.write(f"FP: {best['fp']}\n")
        f.write(f"FN: {best['fn']}\n")

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()