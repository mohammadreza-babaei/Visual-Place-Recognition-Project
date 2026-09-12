"""Adaptive re-ranking using inlier counts."""

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive re-ranking based on inlier threshold."
    )
    parser.add_argument("--preds-dir", required=True)
    parser.add_argument("--inliers-dir", required=True)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        default="adaptive_reranking_outputs"
    )
    return parser.parse_args()


def normalize_path(path):
    return Path(path.strip()).name


def read_prediction_file(pred_file):
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

    return prediction_paths, positive_paths


def load_inliers(inlier_file):
    try:
        results = torch.load(
            inlier_file,
            weights_only=False
        )

        values = []

        for result in results:
            values.append(
                float(result.get("num_inliers", 0))
            )

        return values

    except Exception:
        return []


def main():
    args = parse_args()

    preds_dir = Path(args.preds_dir)
    inliers_dir = Path(args.inliers_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pred_files = sorted(preds_dir.glob("*.txt"))
    inlier_files = sorted(inliers_dir.glob("*.torch"))

    n = min(
        len(pred_files),
        len(inlier_files)
    )

    before_correct = 0
    after_correct = 0
    reranked_queries = 0

    print("\n========== Adaptive Re-ranking ==========")
    print(f"Threshold: {args.threshold}")
    print(f"Prediction files: {len(pred_files)}")
    print(f"Inlier files: {len(inlier_files)}")
    print(f"Processing: {n}")

    for pred_file, inlier_file in zip(
        pred_files[:n],
        inlier_files[:n]
    ):

        prediction_paths, positive_paths = read_prediction_file(
            pred_file
        )

        inliers = load_inliers(inlier_file)

        if not prediction_paths:
            continue

        if not inliers:
            continue

        # R@1 before re-ranking
        original_rank1 = prediction_paths[0]

        if original_rank1 in positive_paths:
            before_correct += 1

        # Default: keep original ranking
        new_predictions = prediction_paths.copy()

        # Inlier count of original R@1
        rank1_inliers = inliers[0]

        # Hard query -> re-rank
        if rank1_inliers < args.threshold:

            reranked_queries += 1

            # Only compare candidates for which we have inlier results
            k = min(
                len(prediction_paths),
                len(inliers)
            )

            best_index = int(
                np.argmax(inliers[:k])
            )

            # Move best candidate to rank 1
            best_prediction = prediction_paths[best_index]

            new_predictions.pop(best_index)
            new_predictions.insert(
                0,
                best_prediction
            )

        # R@1 after re-ranking
        new_rank1 = new_predictions[0]

        if new_rank1 in positive_paths:
            after_correct += 1

        # Save reranked prediction file
        output_file = output_dir / pred_file.name

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("Query path:\n")

            # Recover query path from original file
            original_text = pred_file.read_text(
                errors="ignore"
            )

            query_lines = original_text.splitlines()

            for line in query_lines:
                if line.startswith("Query path:"):
                    f.write(line + "\n")
                    break

            f.write("\nPredictions paths:\n")

            for path in new_predictions:
                f.write(path + "\n")

            f.write("\nPositives paths:\n")

            for path in positive_paths:
                f.write(path + "\n")

    print("\n---------- Results ----------")

    print(f"Queries evaluated: {n}")
    print(f"Before correct:    {before_correct}")
    print(f"After correct:     {after_correct}")
    print(f"Hard queries:      {reranked_queries}")

    if n > 0:
        before_r1 = before_correct / n
        after_r1 = after_correct / n

        print(f"Before R@1:        {before_r1:.4f}")
        print(f"After R@1:         {after_r1:.4f}")
        print(
            f"Improvement:       "
            f"{after_r1 - before_r1:+.4f}"
        )

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()