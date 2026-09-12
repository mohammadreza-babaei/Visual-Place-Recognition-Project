import os
import csv
import argparse


def parse_prediction_file(path):
    predictions = []
    positives = []
    section = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Predictions paths:"):
                section = "predictions"
                continue

            if line.startswith("Positives paths:"):
                section = "positives"
                continue

            if section == "predictions" and line:
                predictions.append(os.path.basename(line))

            elif section == "positives" and line:
                positives.append(os.path.basename(line))

    return predictions, positives


def calculate_r1(directory):
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".txt")
    ]

    if not files:
        return None

    correct = 0

    for path in files:
        predictions, positives = parse_prediction_file(path)

        if predictions and positives:
            if predictions[0] in set(positives):
                correct += 1

    return correct / len(files), len(files)


def find_baseline_predictions(baseline_root, name):

    parts = name.split("_")

    dataset = "_".join(parts[:2])
    method = parts[2]

    search_root = os.path.join(
        baseline_root,
        f"{dataset}_{method}_dot"
    )

    if not os.path.isdir(search_root):
        return None

    candidates = []

    for timestamp in os.listdir(search_root):

        timestamp_path = os.path.join(
            search_root,
            timestamp
        )

        if not os.path.isdir(timestamp_path):
            continue

        preds_path = os.path.join(
            timestamp_path,
            "preds"
        )

        if not os.path.isdir(preds_path):
            continue

        txt_count = sum(
            1
            for f in os.listdir(preds_path)
            if f.lower().endswith(".txt")
        )

        if txt_count > 0:
            candidates.append(
                (txt_count, preds_path)
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return candidates[0][1]


def find_threshold(threshold_root, name):

    path = os.path.join(
        threshold_root,
        name,
        "best_threshold.txt"
    )

    if not os.path.isfile(path):
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:
            line = line.strip()

            if line.startswith("Best threshold:"):
                value = line.split(":", 1)[1].strip()
                return int(value)

    return None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline-root",
        required=True
    )

    parser.add_argument(
        "--adaptive-root",
        required=True
    )

    parser.add_argument(
        "--threshold-root",
        required=True
    )

    parser.add_argument(
        "--output",
        default="adaptive_results.csv"
    )

    args = parser.parse_args()

    rows = []

    adaptive_dirs = sorted([
        d
        for d in os.listdir(args.adaptive_root)
        if os.path.isdir(
            os.path.join(args.adaptive_root, d)
        )
    ])

    print()
    print("Evaluating adaptive re-ranking...")
    print()

    for name in adaptive_dirs:

        adaptive_path = os.path.join(
            args.adaptive_root,
            name
        )

        print(f"Processing: {name}")

        after_result = calculate_r1(
            adaptive_path
        )

        if after_result is None:
            print("  No adaptive prediction files")
            continue

        after_r1, total = after_result

        baseline_path = find_baseline_predictions(
            args.baseline_root,
            name
        )

        if baseline_path is None:
            print("  Baseline not found")
            continue

        before_result = calculate_r1(
            baseline_path
        )

        if before_result is None:
            print("  Baseline has no prediction files")
            continue

        before_r1, baseline_total = before_result

        threshold = find_threshold(
            args.threshold_root,
            name
        )

        improvement = after_r1 - before_r1

        print(f"  Threshold: {threshold}")
        print(f"  Before R@1: {before_r1 * 100:.2f}%")
        print(f"  After R@1:  {after_r1 * 100:.2f}%")
        print(f"  Improvement: {improvement * 100:+.2f} pp")

        rows.append({
            "method": name,
            "threshold": threshold,
            "before_r1": round(before_r1, 6),
            "after_r1": round(after_r1, 6),
            "improvement": round(improvement, 6),
            "total_queries": total,
            "baseline_queries": baseline_total
        })

    output_path = os.path.abspath(
        args.output
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "threshold",
                "before_r1",
                "after_r1",
                "improvement",
                "total_queries",
                "baseline_queries"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 60)
    print(f"Completed: {len(rows)} results")
    print(f"Saved CSV: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()