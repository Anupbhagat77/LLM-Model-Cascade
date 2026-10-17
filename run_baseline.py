import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from config import LARGE_MODEL, MAX_TOKENS_MMLU, MAX_TOKENS_GSM8K
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


def run_baseline(mmlu_size=5, gsm8k_size=5):
    """
    Runs Scenario 1: Baseline.

    Every question is sent directly to the large model.
    """

    os.makedirs("results", exist_ok=True)

    benchmark_df = load_combined_benchmark(
        mmlu_size=mmlu_size,
        gsm8k_size=gsm8k_size
    )

    results = []

    print("\nRunning baseline experiment...")
    print(f"Large model: {LARGE_MODEL}")
    print(f"Total questions: {len(benchmark_df)}")

    for _, row in tqdm(benchmark_df.iterrows(), total=len(benchmark_df)):
        prompt = create_prompt(row)

        model_result = call_model(
            model_name=LARGE_MODEL,
            prompt=prompt,
            max_tokens=get_max_tokens(row["benchmark"])
        )

        evaluation = evaluate_answer(
            row=row,
            model_output=model_result["answer"]
        )

        results.append({
            "scenario": "baseline_large_model_only",
            "question_id": row["question_id"],
            "benchmark": row["benchmark"],
            "subject": row["subject"],
            "question": row["question"],
            "choices": row["choices"],
            "ground_truth": row["ground_truth"],

            "model_used": model_result["model"],
            "raw_answer": model_result["answer"],
            "extracted_answer": evaluation["extracted_answer"],
            "is_correct": evaluation["is_correct"],

            "input_tokens": model_result["input_tokens"],
            "output_tokens": model_result["output_tokens"],
            "total_tokens": model_result["total_tokens"],
            "estimated_cost_usd": model_result["estimated_cost_usd"],
            "estimated_cost_eur": model_result["estimated_cost_eur"],
            "latency_seconds": model_result["latency_seconds"],

            "success": model_result["success"],
            "error": model_result["error"]
        })

    results_df = pd.DataFrame(results)

    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/baseline_results_{timestamp}.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8")

    accuracy = results_df["is_correct"].mean()
    total_tokens = results_df["total_tokens"].sum()
    total_cost_usd = results_df["estimated_cost_usd"].sum()
    total_cost_eur = results_df["estimated_cost_eur"].sum()
    avg_latency = results_df["latency_seconds"].mean()

    print("\nBaseline experiment completed.")
    print(f"Results saved to: {output_path}")
    print("\nSummary:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total tokens: {total_tokens}")
    print(f"Estimated total cost USD: {total_cost_usd:.6f}")
    print(f"Estimated total cost EUR: {total_cost_eur:.6f}")
    print(f"Average latency seconds: {avg_latency:.3f}")

    return results_df


if __name__ == "__main__":
    run_baseline(
        mmlu_size=50,
        gsm8k_size=50
    )