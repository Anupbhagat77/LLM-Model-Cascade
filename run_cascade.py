import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from config import (
    SMALL_MODEL,
    LARGE_MODEL,
    MAX_TOKENS_MMLU,
    MAX_TOKENS_GSM8K
)

from src.benchmark_loader import load_combined_benchmark
from src.prompts import create_mmlu_prompt, create_gsm8k_prompt
from src.model_client import call_model
from src.evaluator import evaluate_mmlu, evaluate_gsm8k


def create_prompt(row):
    """
    Creates the correct prompt depending on the benchmark type.
    """

    if row["benchmark"] == "mmlu":
        return create_mmlu_prompt(
            question=row["question"],
            choices=row["choices"]
        )

    if row["benchmark"] == "gsm8k":
        return create_gsm8k_prompt(
            question=row["question"]
        )

    raise ValueError(f"Unknown benchmark type: {row['benchmark']}")


def evaluate_answer(row, model_output):
    """
    Evaluates model output depending on the benchmark type.
    """

    if row["benchmark"] == "mmlu":
        return evaluate_mmlu(
            model_output=model_output,
            ground_truth=row["ground_truth"]
        )

    if row["benchmark"] == "gsm8k":
        return evaluate_gsm8k(
            model_output=model_output,
            ground_truth=row["ground_truth"]
        )

    raise ValueError(f"Unknown benchmark type: {row['benchmark']}")


def get_max_tokens(benchmark):
    """
    Uses smaller max token limit for MMLU and larger limit for GSM8K.
    """

    if benchmark == "mmlu":
        return MAX_TOKENS_MMLU

    if benchmark == "gsm8k":
        return MAX_TOKENS_GSM8K

    return 256


def run_cascade(mmlu_size=5, gsm8k_size=5):
    """
    Runs Scenario 2: Model Cascade.

    Every question is sent first to the small model.
    If the small model answer is correct according to benchmark exact-match,
    the answer is accepted.
    If the small model answer is incorrect, the question is escalated to
    the large model.
    """

    os.makedirs("results", exist_ok=True)

    benchmark_df = load_combined_benchmark(
        mmlu_size=mmlu_size,
        gsm8k_size=gsm8k_size
    )

    results = []

    print("\nRunning cascade experiment...")
    print(f"Small model: {SMALL_MODEL}")
    print(f"Large model: {LARGE_MODEL}")
    print(f"Total questions: {len(benchmark_df)}")

    for _, row in tqdm(benchmark_df.iterrows(), total=len(benchmark_df)):
        prompt = create_prompt(row)
        max_tokens = get_max_tokens(row["benchmark"])

        small_result = call_model(
            model_name=SMALL_MODEL,
            prompt=prompt,
            max_tokens=max_tokens
        )

        small_evaluation = evaluate_answer(
            row=row,
            model_output=small_result["answer"]
        )

        escalated = False
        large_result = None
        large_evaluation = None

        if small_result["success"] and small_evaluation["is_correct"]:
            final_model_used = small_result["model"]
            final_raw_answer = small_result["answer"]
            final_extracted_answer = small_evaluation["extracted_answer"]
            final_is_correct = small_evaluation["is_correct"]
            routing_decision = "accepted_small_model"

        else:
            escalated = True

            large_result = call_model(
                model_name=LARGE_MODEL,
                prompt=prompt,
                max_tokens=max_tokens
            )

            large_evaluation = evaluate_answer(
                row=row,
                model_output=large_result["answer"]
            )

            final_model_used = large_result["model"]
            final_raw_answer = large_result["answer"]
            final_extracted_answer = large_evaluation["extracted_answer"]
            final_is_correct = large_evaluation["is_correct"]

            if not small_result["success"]:
                routing_decision = "small_model_error_escalated"
            else:
                routing_decision = "small_model_incorrect_escalated"

        if large_result is None:
            large_model_used = None
            large_raw_answer = None
            large_extracted_answer = None
            large_is_correct = None
            large_input_tokens = 0
            large_output_tokens = 0
            large_total_tokens = 0
            large_cost_usd = 0.0
            large_cost_eur = 0.0
            large_latency_seconds = 0.0
            large_success = None
            large_error = None
        else:
            large_model_used = large_result["model"]
            large_raw_answer = large_result["answer"]
            large_extracted_answer = large_evaluation["extracted_answer"]
            large_is_correct = large_evaluation["is_correct"]
            large_input_tokens = large_result["input_tokens"]
            large_output_tokens = large_result["output_tokens"]
            large_total_tokens = large_result["total_tokens"]
            large_cost_usd = large_result["estimated_cost_usd"]
            large_cost_eur = large_result["estimated_cost_eur"]
            large_latency_seconds = large_result["latency_seconds"]
            large_success = large_result["success"]
            large_error = large_result["error"]

        total_tokens = small_result["total_tokens"] + large_total_tokens
        total_cost_usd = small_result["estimated_cost_usd"] + large_cost_usd
        total_cost_eur = small_result["estimated_cost_eur"] + large_cost_eur
        total_latency_seconds = small_result["latency_seconds"] + large_latency_seconds

        results.append({
            "scenario": "cascade_small_then_large",
            "question_id": row["question_id"],
            "benchmark": row["benchmark"],
            "subject": row["subject"],
            "question": row["question"],
            "choices": row["choices"],
            "ground_truth": row["ground_truth"],

            "small_model_used": small_result["model"],
            "small_raw_answer": small_result["answer"],
            "small_extracted_answer": small_evaluation["extracted_answer"],
            "small_is_correct": small_evaluation["is_correct"],
            "small_input_tokens": small_result["input_tokens"],
            "small_output_tokens": small_result["output_tokens"],
            "small_total_tokens": small_result["total_tokens"],
            "small_cost_usd": small_result["estimated_cost_usd"],
            "small_cost_eur": small_result["estimated_cost_eur"],
            "small_latency_seconds": small_result["latency_seconds"],
            "small_success": small_result["success"],
            "small_error": small_result["error"],

            "escalated": escalated,
            "routing_decision": routing_decision,

            "large_model_used": large_model_used,
            "large_raw_answer": large_raw_answer,
            "large_extracted_answer": large_extracted_answer,
            "large_is_correct": large_is_correct,
            "large_input_tokens": large_input_tokens,
            "large_output_tokens": large_output_tokens,
            "large_total_tokens": large_total_tokens,
            "large_cost_usd": large_cost_usd,
            "large_cost_eur": large_cost_eur,
            "large_latency_seconds": large_latency_seconds,
            "large_success": large_success,
            "large_error": large_error,

            "final_model_used": final_model_used,
            "final_raw_answer": final_raw_answer,
            "final_extracted_answer": final_extracted_answer,
            "final_is_correct": final_is_correct,

            "total_tokens": total_tokens,
            "total_cost_usd": total_cost_usd,
            "total_cost_eur": total_cost_eur,
            "total_latency_seconds": total_latency_seconds
        })

    results_df = pd.DataFrame(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/cascade_results_{timestamp}.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8")

    accuracy = results_df["final_is_correct"].mean()
    total_tokens = results_df["total_tokens"].sum()
    total_cost_usd = results_df["total_cost_usd"].sum()
    total_cost_eur = results_df["total_cost_eur"].sum()
    avg_latency = results_df["total_latency_seconds"].mean()

    escalation_rate = results_df["escalated"].mean()
    accepted_by_small = len(results_df[results_df["escalated"] == False])
    escalated_count = len(results_df[results_df["escalated"] == True])

    print("\nCascade experiment completed.")
    print(f"Results saved to: {output_path}")

    print("\nSummary:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total tokens: {total_tokens}")
    print(f"Estimated total cost USD: {total_cost_usd:.6f}")
    print(f"Estimated total cost EUR: {total_cost_eur:.6f}")
    print(f"Average latency seconds: {avg_latency:.3f}")
    print(f"Accepted by small model: {accepted_by_small}")
    print(f"Escalated to large model: {escalated_count}")
    print(f"Escalation rate: {escalation_rate:.2%}")

    print("\nAccuracy by benchmark:")
    print(results_df.groupby("benchmark")["final_is_correct"].mean())

    print("\nEscalation rate by benchmark:")
    print(results_df.groupby("benchmark")["escalated"].mean())

    return results_df


if __name__ == "__main__":
    run_cascade(
        mmlu_size=50,
        gsm8k_size=50
    )