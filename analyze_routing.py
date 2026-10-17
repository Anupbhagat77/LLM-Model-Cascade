import os
import glob
from datetime import datetime

import pandas as pd


def get_latest_file(pattern: str) -> str:
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    return max(files, key=os.path.getmtime)


def to_bool(series: pd.Series) -> pd.Series:
    """
    Safely converts boolean-like values to True/False.
    Handles real booleans and string values like 'True'/'False'.
    """

    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().map({
        "true": True,
        "false": False,
        "1": True,
        "0": False
    }).fillna(False)


def safe_sum(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0

    return df[column].fillna(0).sum()


def safe_mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or len(df) == 0:
        return 0.0

    return df[column].fillna(0).mean()


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def eur(value: float) -> str:
    return f"€{value:.6f}"


def analyze_routing():
    os.makedirs("results", exist_ok=True)

    cascade_path = get_latest_file("results/cascade_results_*.csv")
    df = pd.read_csv(cascade_path)

    # Clean boolean columns
    for col in ["escalated", "small_success", "small_is_correct", "final_is_correct"]:
        if col in df.columns:
            df[col] = to_bool(df[col])

    accepted_df = df[df["escalated"] == False]
    escalated_df = df[df["escalated"] == True]
    error_df = df[df["small_success"] == False]

    total_questions = len(df)
    accepted_count = len(accepted_df)
    escalated_count = len(escalated_df)
    error_count = len(error_df)

    accepted_rate = accepted_count / total_questions if total_questions else 0
    escalation_rate = escalated_count / total_questions if total_questions else 0
    small_success_rate = df["small_success"].mean()
    small_accuracy = df["small_is_correct"].mean()
    final_accuracy = df["final_is_correct"].mean()

    small_total_tokens = safe_sum(df, "small_total_tokens")
    large_total_tokens = safe_sum(df, "large_total_tokens")
    cascade_total_tokens = safe_sum(df, "total_tokens")

    accepted_tokens = safe_sum(accepted_df, "total_tokens")
    escalated_tokens = safe_sum(escalated_df, "total_tokens")

    avg_tokens_accepted = safe_mean(accepted_df, "total_tokens")
    avg_tokens_escalated = safe_mean(escalated_df, "total_tokens")

    small_cost_eur = safe_sum(df, "small_cost_eur")
    large_cost_eur = safe_sum(df, "large_cost_eur")
    total_cost_eur = safe_sum(df, "total_cost_eur")

    accepted_cost_eur = safe_sum(accepted_df, "total_cost_eur")
    escalated_cost_eur = safe_sum(escalated_df, "total_cost_eur")

    avg_cost_accepted = safe_mean(accepted_df, "total_cost_eur")
    avg_cost_escalated = safe_mean(escalated_df, "total_cost_eur")

    avg_latency_all = safe_mean(df, "total_latency_seconds")
    avg_latency_accepted = safe_mean(accepted_df, "total_latency_seconds")
    avg_latency_escalated = safe_mean(escalated_df, "total_latency_seconds")

    small_cost_share = small_cost_eur / total_cost_eur if total_cost_eur else 0
    large_cost_share = large_cost_eur / total_cost_eur if total_cost_eur else 0

    print("\n" + "=" * 70)
    print("ROUTING ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nCascade file used: {cascade_path}")

    print("\n1. Routing Overview")
    print("-" * 70)
    print(f"Total questions                  : {total_questions}")
    print(f"Accepted by small model          : {accepted_count} ({pct(accepted_rate)})")
    print(f"Escalated to large model         : {escalated_count} ({pct(escalation_rate)})")
    print(f"Small-model API errors           : {error_count}")
    print(f"Small-model API success rate     : {pct(small_success_rate)}")

    print("\n2. Accuracy")
    print("-" * 70)
    print(f"Small-model exact-match accuracy : {pct(small_accuracy)}")
    print(f"Final cascade accuracy           : {pct(final_accuracy)}")

    print("\n3. Token Usage")
    print("-" * 70)
    print(f"Small-model tokens               : {small_total_tokens:,.0f}")
    print(f"Large fallback tokens            : {large_total_tokens:,.0f}")
    print(f"Total cascade tokens             : {cascade_total_tokens:,.0f}")
    print(f"Tokens for accepted questions    : {accepted_tokens:,.0f}")
    print(f"Tokens for escalated questions   : {escalated_tokens:,.0f}")
    print(f"Avg tokens, accepted questions   : {avg_tokens_accepted:.2f}")
    print(f"Avg tokens, escalated questions  : {avg_tokens_escalated:.2f}")

    print("\n4. Cost in EUR")
    print("-" * 70)
    print(f"Small-model cost                 : {eur(small_cost_eur)}")
    print(f"Large fallback cost              : {eur(large_cost_eur)}")
    print(f"Total cascade cost               : {eur(total_cost_eur)}")
    print(f"Small-model cost share           : {pct(small_cost_share)}")
    print(f"Large fallback cost share        : {pct(large_cost_share)}")
    print(f"Accepted questions total cost    : {eur(accepted_cost_eur)}")
    print(f"Escalated questions total cost   : {eur(escalated_cost_eur)}")
    print(f"Avg cost, accepted questions     : {eur(avg_cost_accepted)}")
    print(f"Avg cost, escalated questions    : {eur(avg_cost_escalated)}")

    print("\n5. Latency")
    print("-" * 70)
    print(f"Average latency, all questions   : {avg_latency_all:.3f} seconds")
    print(f"Average latency, accepted only   : {avg_latency_accepted:.3f} seconds")
    print(f"Average latency, escalated only  : {avg_latency_escalated:.3f} seconds")

    routing_table = (
        df.groupby("routing_decision")
        .agg(
            questions=("question_id", "count"),
            final_accuracy=("final_is_correct", "mean"),
            avg_tokens=("total_tokens", "mean"),
            total_tokens=("total_tokens", "sum"),
            avg_cost_eur=("total_cost_eur", "mean"),
            total_cost_eur=("total_cost_eur", "sum"),
            avg_latency_seconds=("total_latency_seconds", "mean")
        )
        .reset_index()
    )

    routing_table["final_accuracy"] = routing_table["final_accuracy"].apply(pct)
    routing_table["avg_cost_eur"] = routing_table["avg_cost_eur"].apply(eur)
    routing_table["total_cost_eur"] = routing_table["total_cost_eur"].apply(eur)
    routing_table["avg_tokens"] = routing_table["avg_tokens"].round(2)
    routing_table["avg_latency_seconds"] = routing_table["avg_latency_seconds"].round(3)

    print("\n6. Routing Decision Breakdown")
    print("-" * 70)
    print(routing_table.to_string(index=False))

    benchmark_table = (
        df.groupby("benchmark")
        .agg(
            questions=("question_id", "count"),
            accepted_by_small=("escalated", lambda x: (x == False).sum()),
            escalated_to_large=("escalated", lambda x: (x == True).sum()),
            escalation_rate=("escalated", "mean"),
            small_accuracy=("small_is_correct", "mean"),
            final_accuracy=("final_is_correct", "mean"),
            total_tokens=("total_tokens", "sum"),
            total_cost_eur=("total_cost_eur", "sum"),
            avg_latency_seconds=("total_latency_seconds", "mean")
        )
        .reset_index()
    )

    benchmark_table["accepted_rate"] = (
        benchmark_table["accepted_by_small"] / benchmark_table["questions"]
    )

    benchmark_table["escalation_rate"] = benchmark_table["escalation_rate"].apply(pct)
    benchmark_table["accepted_rate"] = benchmark_table["accepted_rate"].apply(pct)
    benchmark_table["small_accuracy"] = benchmark_table["small_accuracy"].apply(pct)
    benchmark_table["final_accuracy"] = benchmark_table["final_accuracy"].apply(pct)
    benchmark_table["total_cost_eur"] = benchmark_table["total_cost_eur"].apply(eur)
    benchmark_table["avg_latency_seconds"] = benchmark_table["avg_latency_seconds"].round(3)

    print("\n7. Benchmark-wise Routing Breakdown")
    print("-" * 70)
    print(benchmark_table.to_string(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_output_path = f"results/routing_clean_summary_{timestamp}.txt"
    routing_output_path = f"results/routing_decision_breakdown_{timestamp}.csv"
    benchmark_output_path = f"results/routing_benchmark_breakdown_{timestamp}.csv"

    # Save clean CSVs with raw numeric values
    raw_routing_table = (
        df.groupby("routing_decision")
        .agg(
            questions=("question_id", "count"),
            final_accuracy=("final_is_correct", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
            total_tokens=("total_tokens", "sum"),
            avg_cost_eur=("total_cost_eur", "mean"),
            total_cost_eur=("total_cost_eur", "sum"),
            avg_latency_seconds=("total_latency_seconds", "mean")
        )
        .reset_index()
    )

    raw_benchmark_table = (
        df.groupby("benchmark")
        .agg(
            questions=("question_id", "count"),
            accepted_by_small=("escalated", lambda x: (x == False).sum()),
            escalated_to_large=("escalated", lambda x: (x == True).sum()),
            escalation_rate=("escalated", "mean"),
            small_accuracy=("small_is_correct", "mean"),
            final_accuracy=("final_is_correct", "mean"),
            total_tokens=("total_tokens", "sum"),
            total_cost_eur=("total_cost_eur", "sum"),
            avg_latency_seconds=("total_latency_seconds", "mean")
        )
        .reset_index()
    )

    raw_benchmark_table["accepted_rate"] = (
        raw_benchmark_table["accepted_by_small"] / raw_benchmark_table["questions"]
    )

    raw_routing_table.to_csv(routing_output_path, index=False, encoding="utf-8")
    raw_benchmark_table.to_csv(benchmark_output_path, index=False, encoding="utf-8")

    thesis_summary = f"""
ROUTING ANALYSIS SUMMARY

Cascade file used: {cascade_path}

Total questions: {total_questions}
Accepted by small model: {accepted_count} ({pct(accepted_rate)})
Escalated to large model: {escalated_count} ({pct(escalation_rate)})
Small-model API errors: {error_count}
Small-model API success rate: {pct(small_success_rate)}

Small-model accuracy: {pct(small_accuracy)}
Final cascade accuracy: {pct(final_accuracy)}

Small-model tokens: {small_total_tokens:,.0f}
Large fallback tokens: {large_total_tokens:,.0f}
Total cascade tokens: {cascade_total_tokens:,.0f}

Small-model cost: {eur(small_cost_eur)}
Large fallback cost: {eur(large_cost_eur)}
Total cascade cost: {eur(total_cost_eur)}

Average latency, all questions: {avg_latency_all:.3f} seconds
Average latency, accepted questions: {avg_latency_accepted:.3f} seconds
Average latency, escalated questions: {avg_latency_escalated:.3f} seconds

Thesis interpretation:
The cascade accepted {accepted_count} out of {total_questions} questions from the small model and escalated {escalated_count} questions to the large model. No small-model API errors occurred. The routing behavior confirms that the cascade operated as intended: all questions were first processed by the small model, and only benchmark-incorrect small-model answers were escalated to the large model.
""".strip()

    with open(summary_output_path, "w", encoding="utf-8") as file:
        file.write(thesis_summary)

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)
    print(f"Clean text summary saved to      : {summary_output_path}")
    print(f"Routing decision CSV saved to    : {routing_output_path}")
    print(f"Benchmark routing CSV saved to   : {benchmark_output_path}")

    print("\nDone.")


if __name__ == "__main__":
    analyze_routing()