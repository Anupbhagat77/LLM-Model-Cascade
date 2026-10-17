from src.benchmark_loader import load_combined_benchmark_sample


df = load_combined_benchmark_sample(
    mmlu_size=5,
    gsm8k_size=5
)

print("\nLoaded benchmark sample:")
print(df)

print("\nColumns:")
print(df.columns.tolist())

print("\nBenchmark counts:")
print(df["benchmark"].value_counts())

print("\nFirst row:")
print(df.iloc[0].to_dict())

print("\nBenchmark loader test completed.")