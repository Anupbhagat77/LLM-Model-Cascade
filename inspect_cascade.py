import glob
import os
import pandas as pd

latest_file = max(
    glob.glob("results/cascade_results_*.csv"),
    key=os.path.getmtime
)

print(f"Inspecting latest cascade file: {latest_file}")

df = pd.read_csv(latest_file)

print("\nTotal questions:")
print(len(df))

print("\nSmall model success count:")
print(df["small_success"].value_counts(dropna=False))

print("\nRouting decision counts:")
print(df["routing_decision"].value_counts(dropna=False))

print("\nEscalation count:")
print(df["escalated"].value_counts(dropna=False))

print("\nSmall model errors:")
print(df[df["small_success"] == False][["question_id", "benchmark", "small_error"]])

print("\nToken summary:")
print(df[[
    "question_id",
    "benchmark",
    "small_total_tokens",
    "large_total_tokens",
    "total_tokens",
    "escalated",
    "routing_decision"
]].to_string(index=False))