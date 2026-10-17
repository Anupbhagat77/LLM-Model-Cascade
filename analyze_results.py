import os
import glob
from datetime import datetime

import pandas as pd


def get_latest_file(pattern: str) -> str:
    """
    Finds the latest file matching a pattern.
    Example:
    results/baseline_results_*.csv
    """

    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def calculate_percentage_change(baseline_value, cascade_value):
    """
    Calculates percentage change from baseline to cascade.
    Positive value means reduction/improvement if baseline is larger.
    Negative value means increase/worse if cascade is larger.
    """

    if baseline_value == 0:
        return 0

    return ((baseline_value - cascade_value) / baseline_value) * 100


def analyze_results():
    """
    Compares latest baseline and cascade result files.
    """

    os.makedirs("results", exist_ok=True)

    baseline_path = get_latest_file("results/baseline_results_*.csv")
    cascade_path = get_latest_file("results/cascade_results_*.csv")

    print("\nUsing baseline file:")
    print(baseline_path)

    print("\nUsing cascade file:")
    print(cascade_path)

    baseline_df = pd.read_csv(baseline_path)
    cascade_df = pd.read_csv(cascade_path)

    # Overall baseline metrics
    baseline_accuracy = baseline_df["is_correct"].mean()
    baseline_total_tokens = baseline_df["total_tokens"].sum()
    baseline_total_cost_usd = baseline_df["estimated_cost_usd"].sum()
    baseline_total_cost_eur = baseline_df["estimated_cost_eur"].sum()
    baseline_avg_latency = baseline_df["latency_seconds"].mean()

    # Overall cascade metrics
    cascade_accuracy = cascade_df["final_is_correct"].mean()
    cascade_total_tokens = cascade_df["total_tokens"].sum()
    cascade_total_cost_usd = cascade_df["total_cost_usd"].sum()
    cascade_total_cost_eur = cascade_df["total_cost_eur"].sum()
    cascade_avg_latency = cascade_df["total_latency_seconds"].mean()
    escalation_rate = cascade_df["escalated"].mean()

    # Derived metrics
    if baseline_accuracy == 0:
        quality_retention = 0
    else:
        quality_retention = cascade_accuracy / baseline_accuracy

    token_reduction_percent = calculate_percentage_change(
        baseline_total_tokens,
        cascade_total_tokens
    )

    cost_reduction_percent_usd = calculate_percentage_change(
        baseline_total_cost_usd,
        cascade_total_cost_usd
    )

    cost_reduction_percent_eur = calculate_percentage_change(
        baseline_total_cost_eur,
        cascade_total_cost_eur
    )

    latency_improvement_percent = calculate_percentage_change(
        baseline_avg_latency,
        cascade_avg_latency
    )

    comparison_rows = [
        {
            "metric": "accuracy",
            "baseline": baseline_accuracy,
            "cascade": cascade_accuracy,
            "change": cascade_accuracy - baseline_accuracy,
            "interpretation": "Higher is better"
        },
        {
            "metric": "quality_retention",
            "baseline": 1.0,
            "cascade": quality_retention,
            "change": quality_retention - 1.0,
            "interpretation": "Cascade should be at least 0.95"
        },
        {
            "metric": "total_tokens",
            "baseline": baseline_total_tokens,
            "cascade": cascade_total_tokens,
            "change": cascade_total_tokens - baseline_total_tokens,
            "interpretation": "Lower is better"
        },
        {
            "metric": "token_reduction_percent",
            "baseline": 0,
            "cascade": token_reduction_percent,
            "change": token_reduction_percent,
            "interpretation": "Positive means token reduction, negative means token increase"
        },
        {
            "metric": "total_cost_usd",
            "baseline": baseline_total_cost_usd,
            "cascade": cascade_total_cost_usd,
            "change": cascade_total_cost_usd - baseline_total_cost_usd,
            "interpretation": "Lower is better"
        },
        {
            "metric": "total_cost_eur",
            "baseline": baseline_total_cost_eur,
            "cascade": cascade_total_cost_eur,
            "change": cascade_total_cost_eur - baseline_total_cost_eur,
            "interpretation": "Lower is better"
        },
        {
            "metric": "cost_reduction_percent_eur",
            "baseline": 0,
            "cascade": cost_reduction_percent_eur,
            "change": cost_reduction_percent_eur,
            "interpretation": "Positive means cost reduction"
        },
        {
            "metric": "average_latency_seconds",
            "baseline": baseline_avg_latency,
            "cascade": cascade_avg_latency,
            "change": cascade_avg_latency - baseline_avg_latency,
            "interpretation": "Lower is better"
        },
        {
            "metric": "latency_improvement_percent",
            "baseline": 0,
            "cascade": latency_improvement_percent,
            "change": latency_improvement_percent,
            "interpretation": "Positive means latency improved"
        },
        {
            "metric": "escalation_rate",
            "baseline": None,
            "cascade": escalation_rate,
            "change": None,
            "interpretation": "Percentage of queries escalated to the large model"
        }
    ]

    comparison_df = pd.DataFrame(comparison_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_output_path = f"results/final_comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_output_path, index=False, encoding="utf-8")

    print("\nOverall comparison:")
    print(comparison_df.to_string(index=False))

    print("\nBenchmark-wise accuracy:")
    baseline_accuracy_by_benchmark = baseline_df.groupby("benchmark")["is_correct"].mean()
    cascade_accuracy_by_benchmark = cascade_df.groupby("benchmark")["final_is_correct"].mean()

    benchmark_comparison = pd.DataFrame({
        "baseline_accuracy": baseline_accuracy_by_benchmark,
        "cascade_accuracy": cascade_accuracy_by_benchmark
    })

    benchmark_comparison["quality_retention"] = (
        benchmark_comparison["cascade_accuracy"] / benchmark_comparison["baseline_accuracy"]
    )

    print(benchmark_comparison)

    benchmark_output_path = f"results/benchmark_comparison_{timestamp}.csv"
    benchmark_comparison.to_csv(benchmark_output_path, encoding="utf-8")

    print("\nAnalysis completed.")
    print(f"Overall comparison saved to: {comparison_output_path}")
    print(f"Benchmark comparison saved to: {benchmark_output_path}")

    print("\nKey interpretation:")
    print(f"Baseline accuracy: {baseline_accuracy:.2%}")
    print(f"Cascade accuracy: {cascade_accuracy:.2%}")
    print(f"Quality retention: {quality_retention:.2%}")
    print(f"Token reduction: {token_reduction_percent:.2f}%")
    print(f"Cost reduction EUR: {cost_reduction_percent_eur:.2f}%")
    print(f"Latency improvement: {latency_improvement_percent:.2f}%")
    print(f"Escalation rate: {escalation_rate:.2%}")


if __name__ == "__main__":
    analyze_results()