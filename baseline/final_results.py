import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("adaptive_results_final.xlsx")

df["Label"] = (
    df["Dataset"] + " | "
    + df["Method"] + " | "
    + df["Matcher"]
)

x = range(len(df))

plt.figure(figsize=(14, 8))

plt.plot(
    x,
    df["Before R@1 (%)"],
    marker="o",
    label="Before Adaptive Re-ranking",
)

plt.plot(
    x,
    df["After R@1 (%)"],
    marker="o",
    label="After Adaptive Re-ranking",
)

plt.xticks(x, df["Label"], rotation=90)
plt.ylabel("R@1 (%)")
plt.title("Adaptive Re-ranking: Before vs After R@1")
plt.legend()
plt.tight_layout()

plt.savefig("adaptive_before_after.png", dpi=300)
plt.close()

print("Saved: adaptive_before_after.png")

print("\nBest improvements:")
print(
    df.sort_values("Improvement (pp)", ascending=False)
    [["Dataset", "Method", "Matcher", "Improvement (pp)"]]
    .to_string(index=False)
)

plt.figure(figsize=(14, 8))

sorted_df = df.sort_values("Improvement (pp)", ascending=True)

labels = (
    sorted_df["Dataset"] + " | "
    + sorted_df["Method"] + " | "
    + sorted_df["Matcher"]
)

plt.barh(
    labels,
    sorted_df["Improvement (pp)"]
)

plt.xlabel("R@1 Improvement (percentage points)")
plt.title("Adaptive Re-ranking Improvement")
plt.tight_layout()

plt.savefig("adaptive_improvement.png", dpi=300)
plt.close()

print("Saved: adaptive_improvement.png")

final_columns = [
    "Dataset",
    "Method",
    "Matcher",
    "Threshold",
    "Before R@1 (%)",
    "After R@1 (%)",
    "Improvement (pp)",
    "Queries",
]

final_df = df[final_columns].copy()

final_df.to_excel(
    "adaptive_results_clean.xlsx",
    index=False,
)

final_df.to_csv(
    "adaptive_results_clean.csv",
    index=False,
)

best_df = (
    final_df
    .sort_values("Improvement (pp)", ascending=False)
    .groupby(["Dataset", "Method"], as_index=False)
    .first()
)

best_df.to_excel(
    "adaptive_best_results.xlsx",
    index=False,
)

print("Saved: adaptive_results_clean.xlsx")
print("Saved: adaptive_results_clean.csv")
print("Saved: adaptive_best_results.xlsx")