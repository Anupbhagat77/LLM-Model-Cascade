from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd


BASELINE_PATH = Path("results/baseline_results_20260614_233141.csv")
CASCADE_PATH = Path("results/cascade_results_20260614_233435.csv")
RESULTS_DIR = Path("results")

THRESHOLDS = [0.70, 0.90]


def latest_judge_file():
    files = sorted(RESULTS_DIR.glob("judge_results_*.csv"))
    if not files:
        raise FileNotFoundError("No judge_results_*.csv file found in results folder.")
    return files[-1]


def to_bool(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def scenario_row(name, accuracy, total_cost_eur, total_cost_usd, total_tokens, avg_latency, baseline_accuracy, baseline_cost_eur, baseline_tokens, escalation_rate=None):
    return {
        "scenario": name,
        "accuracy": accuracy,
        "quality_retention": accuracy / baseline_accuracy if baseline_accuracy else np.nan,
        "total_cost_eur": total_cost_eur,
        "total_cost_usd": total_cost_usd,
        "cost_reduction_vs_gpt41": 1 - (total_cost_eur / baseline_cost_eur) if baseline_cost_eur else np.nan,
        "total_tokens": total_tokens,
        "token_change_vs_gpt41": (total_tokens / baseline_tokens) - 1 if baseline_tokens else np.nan,
        "avg_latency_seconds": avg_latency,
        "escalation_rate": escalation_rate,
    }


def main():
    if len(sys.argv) > 1:
        judge_path = Path(sys.argv[1])
    else:
        judge_path = latest_judge_file()

    print(f"Using judge file: {judge_path}")

    baseline = pd.read_csv(BASELINE_PATH)
    cascade = pd.read_csv(CASCADE_PATH)
    judge = pd.read_csv(judge_path)

    baseline_correct = to_bool(baseline["is_correct"])
    cascade_final_correct = to_bool(cascade["final_is_correct"])
    small_correct = to_bool(cascade["small_is_correct"])

    baseline_accuracy = baseline_correct.mean()
    baseline_cost_eur = baseline["estimated_cost_eur"].sum()
    baseline_cost_usd = baseline["estimated_cost_usd"].sum()
    baseline_tokens = baseline["total_tokens"].sum()
    baseline_latency = baseline["latency_seconds"].mean()

    small_accuracy = small_correct.mean()
    small_cost_eur = cascade["small_cost_eur"].sum()
    small_cost_usd = cascade["small_cost_usd"].sum()
    small_tokens = cascade["small_total_tokens"].sum()
    small_latency = cascade["small_latency_seconds"].mean()

    oracle_accuracy = cascade_final_correct.mean()
    oracle_cost_eur = cascade["total_cost_eur"].sum()
    oracle_cost_usd = cascade["total_cost_usd"].sum()
    oracle_tokens = cascade["total_tokens"].sum()
    oracle_latency = cascade["total_latency_seconds"].mean()
    oracle_escalation_rate = to_bool(cascade["escalated"]).mean()

    rows = []

    rows.append(
        scenario_row(
            name="GPT-4.1 only baseline",
            accuracy=baseline_accuracy,
            total_cost_eur=baseline_cost_eur,
            total_cost_usd=baseline_cost_usd,
            total_tokens=baseline_tokens,
            avg_latency=baseline_latency,
            baseline_accuracy=baseline_accuracy,
            baseline_cost_eur=baseline_cost_eur,
            baseline_tokens=baseline_tokens,
            escalation_rate=np.nan,
        )
    )

    rows.append(
        scenario_row(
            name="Gemini only baseline",
            accuracy=small_accuracy,
            total_cost_eur=small_cost_eur,
            total_cost_usd=small_cost_usd,
            total_tokens=small_tokens,
            avg_latency=small_latency,
            baseline_accuracy=baseline_accuracy,
            baseline_cost_eur=baseline_cost_eur,
            baseline_tokens=baseline_tokens,
            escalation_rate=0.0,
        )
    )

    rows.append(
        scenario_row(
            name="Oracle cascade",
            accuracy=oracle_accuracy,
            total_cost_eur=oracle_cost_eur,
            total_cost_usd=oracle_cost_usd,
            total_tokens=oracle_tokens,
            avg_latency=oracle_latency,
            baseline_accuracy=baseline_accuracy,
            baseline_cost_eur=baseline_cost_eur,
            baseline_tokens=baseline_tokens,
            escalation_rate=oracle_escalation_rate,
        )
    )

    router_error_rows = []

    for threshold in THRESHOLDS:
        judge_decision = judge["judge_decision"].astype(str).str.upper()
        judge_confidence = pd.to_numeric(judge["judge_confidence"], errors="coerce").fillna(0.0)
        judge_success = to_bool(judge["judge_success"])
        judge_parse_success = to_bool(judge["judge_parse_success"])

        judge_accept = (
            (judge_decision == "ACCEPT")
            & (judge_confidence >= threshold)
            & judge_success
            & judge_parse_success
        )

        judge_escalate = ~judge_accept

        judge_small_correct = to_bool(judge["small_is_correct"])
        judge_baseline_correct = to_bool(judge["baseline_is_correct"])

        final_correct = np.where(judge_accept, judge_small_correct, judge_baseline_correct)

        total_cost_eur = (
            judge["small_cost_eur"]
            + judge["judge_cost_eur"]
            + np.where(judge_accept, 0.0, judge["baseline_cost_eur"])
        ).sum()

        total_cost_usd = (
            judge["small_cost_usd"]
            + judge["judge_cost_usd"]
            + np.where(judge_accept, 0.0, judge["baseline_cost_usd"])
        ).sum()

        total_tokens = (
            judge["small_total_tokens"]
            + judge["judge_total_tokens"]
            + np.where(judge_accept, 0, judge["baseline_total_tokens"])
        ).sum()

        total_latency = (
            judge["small_latency_seconds"]
            + judge["judge_latency_seconds"]
            + np.where(judge_accept, 0.0, judge["baseline_latency_seconds"])
        )

        accuracy = final_correct.mean()
        avg_latency = total_latency.mean()
        escalation_rate = judge_escalate.mean()

        scenario_name = f"LLM-as-judge cascade, threshold {threshold:.2f}"

        rows.append(
            scenario_row(
                name=scenario_name,
                accuracy=accuracy,
                total_cost_eur=total_cost_eur,
                total_cost_usd=total_cost_usd,
                total_tokens=total_tokens,
                avg_latency=avg_latency,
                baseline_accuracy=baseline_accuracy,
                baseline_cost_eur=baseline_cost_eur,
                baseline_tokens=baseline_tokens,
                escalation_rate=escalation_rate,
            )
        )

        true_accept = (judge_accept & judge_small_correct).sum()
        false_accept = (judge_accept & ~judge_small_correct).sum()
        true_escalate = (judge_escalate & ~judge_small_correct).sum()
        false_escalate = (judge_escalate & judge_small_correct).sum()

        accepted_count = judge_accept.sum()
        escalated_count = judge_escalate.sum()

        router_error_rows.append(
            {
                "threshold": threshold,
                "true_accept": true_accept,
                "false_accept": false_accept,
                "true_escalate": true_escalate,
                "false_escalate": false_escalate,
                "accepted_count": accepted_count,
                "escalated_count": escalated_count,
                "false_accept_rate_total": false_accept / len(judge),
                "false_escalation_rate_total": false_escalate / len(judge),
                "wrong_among_accepted": false_accept / accepted_count if accepted_count else np.nan,
                "unnecessary_among_escalated": false_escalate / escalated_count if escalated_count else np.nan,
            }
        )

    summary = pd.DataFrame(rows)
    router_errors = pd.DataFrame(router_error_rows)

    percent_cols = [
        "accuracy",
        "quality_retention",
        "cost_reduction_vs_gpt41",
        "token_change_vs_gpt41",
        "escalation_rate",
    ]

    display_summary = summary.copy()
    for col in percent_cols:
        display_summary[col] = display_summary[col].apply(
            lambda x: "" if pd.isna(x) else f"{x * 100:.2f}%"
        )

    display_summary["total_cost_eur"] = display_summary["total_cost_eur"].apply(lambda x: f"€{x:.6f}")
    display_summary["total_cost_usd"] = display_summary["total_cost_usd"].apply(lambda x: f"${x:.6f}")
    display_summary["total_tokens"] = display_summary["total_tokens"].apply(lambda x: f"{int(x):,}")
    display_summary["avg_latency_seconds"] = display_summary["avg_latency_seconds"].apply(lambda x: f"{x:.3f}s")

    print("\nSCENARIO COMPARISON")
    print(display_summary.to_string(index=False))

    print("\nROUTER ERROR ANALYSIS")
    print(router_errors.to_string(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = RESULTS_DIR / f"judge_scenario_comparison_{timestamp}.csv"
    router_path = RESULTS_DIR / f"judge_router_errors_{timestamp}.csv"

    summary.to_csv(summary_path, index=False)
    router_errors.to_csv(router_path, index=False)

    print(f"\nSaved scenario comparison to: {summary_path}")
    print(f"Saved router error analysis to: {router_path}")


if __name__ == "__main__":
    main()