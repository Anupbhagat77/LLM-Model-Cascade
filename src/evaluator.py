import re
from decimal import Decimal, InvalidOperation
from typing import Optional


def extract_mmlu_answer(model_output: str) -> Optional[str]:
    """
    Extracts the final answer letter A, B, C, or D from model output.
    """

    if model_output is None:
        return None

    text = model_output.strip().upper()

    final_answer_match = re.search(r"FINAL\s*ANSWER\s*[:\-]?\s*([ABCD])", text)
    if final_answer_match:
        return final_answer_match.group(1)

    first_letter_match = re.search(r"\b([ABCD])\b", text)
    if first_letter_match:
        return first_letter_match.group(1)

    return None


def extract_gsm8k_answer(model_output: str) -> Optional[str]:
    """
    Extracts the final numeric answer from model output.
    Handles formats like:
    Final answer: 42
    #### 42
    The answer is 42.
    """

    if model_output is None:
        return None

    text = model_output.strip()

    final_answer_match = re.search(
        r"FINAL\s*ANSWER\s*[:\-]?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )
    if final_answer_match:
        return final_answer_match.group(1).replace(",", "")

    hash_answer_match = re.search(
        r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        text
    )
    if hash_answer_match:
        return hash_answer_match.group(1).replace(",", "")

    all_numbers = re.findall(
        r"-?\d+(?:,\d{3})*(?:\.\d+)?",
        text
    )

    if all_numbers:
        return all_numbers[-1].replace(",", "")

    return None


def normalize_number(value: str) -> Optional[Decimal]:
    """
    Converts a numeric string into Decimal for fair comparison.
    Example:
    42 and 42.0 should be treated as equal.
    """

    if value is None:
        return None

    try:
        return Decimal(str(value).replace(",", "").strip())
    except InvalidOperation:
        return None


def evaluate_mmlu(model_output: str, ground_truth: str) -> dict:
    """
    Evaluates MMLU answer using exact match.
    """

    extracted_answer = extract_mmlu_answer(model_output)

    if ground_truth is None:
        correct_answer = None
    else:
        correct_answer = str(ground_truth).strip().upper()

    is_correct = extracted_answer == correct_answer

    return {
        "extracted_answer": extracted_answer,
        "ground_truth": correct_answer,
        "is_correct": is_correct
    }


def evaluate_gsm8k(model_output: str, ground_truth: str) -> dict:
    """
    Evaluates GSM8K answer using exact numeric match.
    """

    extracted_answer = extract_gsm8k_answer(model_output)
    correct_answer = extract_gsm8k_answer(str(ground_truth))

    extracted_number = normalize_number(extracted_answer)
    correct_number = normalize_number(correct_answer)

    is_correct = (
        extracted_number is not None
        and correct_number is not None
        and extracted_number == correct_number
    )

    return {
        "extracted_answer": extracted_answer,
        "ground_truth": correct_answer,
        "is_correct": is_correct
    }