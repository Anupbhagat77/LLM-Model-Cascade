import pandas as pd
from datasets import load_dataset

from config import RANDOM_SEED

def load_mmlu_sample(sample_size: int = 25, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Loads a random sample from the MMLU benchmark.

    MMLU answer labels are stored as indices:
    0 = A, 1 = B, 2 = C, 3 = D
    """

    dataset = load_dataset("cais/mmlu", "all", split="test")

    if sample_size > len(dataset):
        sample_size = len(dataset)

    dataset = dataset.shuffle(seed=random_seed).select(range(sample_size))

    rows = []
    answer_map = ["A", "B", "C", "D"]

    for idx, item in enumerate(dataset, start=1):
        rows.append(
            {
                "question_id": f"mmlu_{idx}",
                "benchmark": "mmlu",
                "subject": item.get("subject", "unknown"),
                "question": item["question"],
                "choices": item["choices"],
                "ground_truth": answer_map[item["answer"]],
            }
        )

    return pd.DataFrame(rows)


def extract_gsm8k_answer(answer_text: str) -> str:
    """
    Extracts final answer from GSM8K format.
    GSM8K answers usually end with: #### final_answer
    """

    if "####" in answer_text:
        final_answer = answer_text.split("####")[-1].strip()
    else:
        final_answer = answer_text.strip()

    final_answer = final_answer.replace(",", "")
    return final_answer


def load_gsm8k_sample(sample_size: int = 25, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Loads a random sample from the GSM8K benchmark.
    """

    dataset = load_dataset("openai/gsm8k", "main", split="test")

    if sample_size > len(dataset):
        sample_size = len(dataset)

    dataset = dataset.shuffle(seed=random_seed).select(range(sample_size))

    rows = []

    for idx, item in enumerate(dataset, start=1):
        rows.append(
            {
                "question_id": f"gsm8k_{idx}",
                "benchmark": "gsm8k",
                "subject": "grade_school_math",
                "question": item["question"],
                "choices": None,
                "ground_truth": extract_gsm8k_answer(item["answer"]),
            }
        )

    return pd.DataFrame(rows)


def load_combined_benchmark(
    mmlu_size: int = 25,
    gsm8k_size: int = 25,
    random_seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Loads a combined random sample of MMLU and GSM8K.

    The same seed is used in baseline and cascade scripts,
    so both experiments evaluate exactly the same questions.
    """

    mmlu_df = load_mmlu_sample(
        sample_size=mmlu_size,
        random_seed=random_seed
    )

    gsm8k_df = load_gsm8k_sample(
        sample_size=gsm8k_size,
        random_seed=random_seed
    )

    combined_df = pd.concat([mmlu_df, gsm8k_df], ignore_index=True)

    return combined_df