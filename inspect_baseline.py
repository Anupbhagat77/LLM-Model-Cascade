import pandas as pd

file_path = "results/baseline_results_20260613_182813.csv"

df = pd.read_csv(file_path)

cols = [
    "question_id",
    "benchmark",
    "ground_truth",
    "raw_answer",
    "extracted_answer",
    "is_correct",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "estimated_cost_usd",
    "estimated_cost_eur",
    "latency_seconds"
]

print(df[cols].to_string(index=False))

print("\nAccuracy by benchmark:")
print(df.groupby("benchmark")["is_correct"].mean())

print("\nToken usage by benchmark:")
print(df.groupby("benchmark")["total_tokens"].sum())

print("\nAverage latency by benchmark:")
print(df.groupby("benchmark")["latency_seconds"].mean())