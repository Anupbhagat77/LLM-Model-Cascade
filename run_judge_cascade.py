import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from litellm import completion

try:
    from litellm import completion_cost
except Exception:
    completion_cost = None


BASELINE_PATH = Path("results/baseline_results_20260614_233141.csv")
CASCADE_PATH = Path("results/cascade_results_20260614_233435.csv")
OUTPUT_DIR = Path("results")

JUDGE_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_TOKENS_JUDGE = 180
USD_TO_EUR_RATE = 1 / 1.1567


def get_response_text(response):
    try:
        return response.choices[0].message.content
    except Exception:
        return response["choices"][0]["message"]["content"]


def get_usage_value(response, key):
    try:
        usage = response.usage
        return int(getattr(usage, key, 0) or 0)
    except Exception:
        try:
            return int(response["usage"].get(key, 0) or 0)
        except Exception:
            return 0


def estimate_cost_usd(response, input_tokens, output_tokens):
    if completion_cost is not None:
        try:
            return float(completion_cost(completion_response=response))
        except Exception:
            pass

    # Fallback estimate for GPT-4o-mini:
    # input: $0.15 / 1M tokens, output: $0.60 / 1M tokens
    return (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)


def extract_json_object(text):
    if text is None:
        raise ValueError("Empty judge response")

    cleaned = text.strip()
    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)

    return json.loads(cleaned)


def parse_judge_output(text):
    try:
        obj = extract_json_object(text)

        decision = str(obj.get("decision", "ESCALATE")).strip().upper()
        if decision not in ["ACCEPT", "ESCALATE"]:
            decision = "ESCALATE"

        try:
            confidence = float(obj.get("confidence", 0.0))
        except Exception:
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))
        reason = str(obj.get("reason", "")).strip()

        return {
            "judge_parse_success": True,
            "judge_decision": decision,
            "judge_confidence": confidence,
            "judge_reason": reason,
            "judge_parse_error": "",
        }

    except Exception as e:
        return {
            "judge_parse_success": False,
            "judge_decision": "ESCALATE",
            "judge_confidence": 0.0,
            "judge_reason": "Could not parse judge output. Defaulted to ESCALATE.",
            "judge_parse_error": str(e),
        }


def build_judge_prompt(row):
    benchmark = str(row["benchmark"]).lower()
    question = row["question"]
    choices = row.get("choices", "")
    small_raw_answer = row.get("small_raw_answer", "")
    small_extracted_answer = row.get("small_extracted_answer", "")

    if benchmark == "mmlu":
        return f"""
You are an answer-quality judge in a model cascade system.

Your task is to decide whether a candidate answer from a small model is reliable enough to return to a user, or whether the question should be escalated to a stronger model.

Important:
- You are NOT given the ground-truth answer.
- Use only the question, answer options, and the candidate answer.
- Do not assume the candidate answer is correct.
- For multiple-choice questions, solve/check the question yourself and evaluate whether the extracted letter is the best option.

Question:
{question}

Answer options:
{choices}

Candidate answer from Gemini 2.5 Flash Lite:
Raw answer: {small_raw_answer}
Extracted answer: {small_extracted_answer}

Return JSON only in exactly this format:
{{
  "decision": "ACCEPT" or "ESCALATE",
  "confidence": 0.0,
  "reason": "one short sentence"
}}

Decision rules:
- Use "ACCEPT" only if the candidate answer is clearly correct and properly formatted.
- Use "ESCALATE" if the answer appears wrong, incomplete, unsupported, badly formatted, or uncertain.
""".strip()

    return f"""
You are an answer-quality judge in a model cascade system.

Your task is to decide whether a candidate answer from a small model is reliable enough to return to a user, or whether the question should be escalated to a stronger model.

Important:
- You are NOT given the ground-truth answer.
- Use only the math problem and the candidate answer.
- Do not assume the candidate answer is correct.
- Check whether the reasoning and final numeric answer appear correct.

Math problem:
{question}

Candidate answer from Gemini 2.5 Flash Lite:
Raw answer: {small_raw_answer}
Extracted answer: {small_extracted_answer}

Return JSON only in exactly this format:
{{
  "decision": "ACCEPT" or "ESCALATE",
  "confidence": 0.0,
  "reason": "one short sentence"
}}

Decision rules:
- Use "ACCEPT" only if the candidate answer is clearly correct and properly formatted.
- Use "ESCALATE" if the answer appears wrong, incomplete, unsupported, badly formatted, or uncertain.
""".strip()


def call_judge(prompt, max_retries=3):
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            start = time.perf_counter()

            response = completion(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict answer-quality judge. Return valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS_JUDGE,
            )

            latency = time.perf_counter() - start
            raw_text = get_response_text(response)

            input_tokens = get_usage_value(response, "prompt_tokens")
            output_tokens = get_usage_value(response, "completion_tokens")
            total_tokens = input_tokens + output_tokens

            cost_usd = estimate_cost_usd(response, input_tokens, output_tokens)
            cost_eur = cost_usd * USD_TO_EUR_RATE

            parsed = parse_judge_output(raw_text)

            return {
                "judge_model": JUDGE_MODEL,
                "judge_raw_output": raw_text,
                "judge_input_tokens": input_tokens,
                "judge_output_tokens": output_tokens,
                "judge_total_tokens": total_tokens,
                "judge_cost_usd": cost_usd,
                "judge_cost_eur": cost_eur,
                "judge_latency_seconds": latency,
                "judge_success": True,
                "judge_api_error": "",
                **parsed,
            }

        except Exception as e:
            last_error = str(e)
            print(f"Judge call failed on attempt {attempt}/{max_retries}: {last_error}")
            time.sleep(2 * attempt)

    return {
        "judge_model": JUDGE_MODEL,
        "judge_raw_output": "",
        "judge_input_tokens": 0,
        "judge_output_tokens": 0,
        "judge_total_tokens": 0,
        "judge_cost_usd": 0.0,
        "judge_cost_eur": 0.0,
        "judge_latency_seconds": 0.0,
        "judge_success": False,
        "judge_api_error": last_error,
        "judge_parse_success": False,
        "judge_decision": "ESCALATE",
        "judge_confidence": 0.0,
        "judge_reason": "Judge API failed. Defaulted to ESCALATE.",
        "judge_parse_error": last_error,
    }


def main():
    load_dotenv()
    OUTPUT_DIR.mkdir(exist_ok=True)

    baseline = pd.read_csv(BASELINE_PATH)
    cascade = pd.read_csv(CASCADE_PATH)

    baseline_cols = [
        "question_id",
        "model_used",
        "raw_answer",
        "extracted_answer",
        "is_correct",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "estimated_cost_eur",
        "latency_seconds",
        "success",
        "error",
    ]

    baseline_prefixed = baseline[baseline_cols].rename(
        columns={
            "model_used": "baseline_model_used",
            "raw_answer": "baseline_raw_answer",
            "extracted_answer": "baseline_extracted_answer",
            "is_correct": "baseline_is_correct",
            "input_tokens": "baseline_input_tokens",
            "output_tokens": "baseline_output_tokens",
            "total_tokens": "baseline_total_tokens",
            "estimated_cost_usd": "baseline_cost_usd",
            "estimated_cost_eur": "baseline_cost_eur",
            "latency_seconds": "baseline_latency_seconds",
            "success": "baseline_success",
            "error": "baseline_error",
        }
    )

    data = cascade.merge(baseline_prefixed, on="question_id", how="left")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"judge_results_{timestamp}.csv"

    rows = []

    print(f"Loaded {len(data)} questions.")
    print("Starting LLM-as-judge calls...")
    print(f"Output will be saved to: {output_path}")

    for idx, row in data.iterrows():
        print(f"\nJudging {idx + 1}/{len(data)}: {row['question_id']} ({row['benchmark']})")

        prompt = build_judge_prompt(row)
        judge_result = call_judge(prompt)

        oracle_decision = "ACCEPT" if bool(row["small_is_correct"]) else "ESCALATE"

        out = {
            "scenario": "llm_as_judge_cascade",
            "question_id": row["question_id"],
            "benchmark": row["benchmark"],
            "subject": row.get("subject", ""),
            "question": row["question"],
            "choices": row.get("choices", ""),
            "ground_truth": row["ground_truth"],

            "small_model_used": row["small_model_used"],
            "small_raw_answer": row["small_raw_answer"],
            "small_extracted_answer": row["small_extracted_answer"],
            "small_is_correct": row["small_is_correct"],
            "small_input_tokens": row["small_input_tokens"],
            "small_output_tokens": row["small_output_tokens"],
            "small_total_tokens": row["small_total_tokens"],
            "small_cost_usd": row["small_cost_usd"],
            "small_cost_eur": row["small_cost_eur"],
            "small_latency_seconds": row["small_latency_seconds"],

            "baseline_model_used": row["baseline_model_used"],
            "baseline_raw_answer": row["baseline_raw_answer"],
            "baseline_extracted_answer": row["baseline_extracted_answer"],
            "baseline_is_correct": row["baseline_is_correct"],
            "baseline_input_tokens": row["baseline_input_tokens"],
            "baseline_output_tokens": row["baseline_output_tokens"],
            "baseline_total_tokens": row["baseline_total_tokens"],
            "baseline_cost_usd": row["baseline_cost_usd"],
            "baseline_cost_eur": row["baseline_cost_eur"],
            "baseline_latency_seconds": row["baseline_latency_seconds"],

            "oracle_decision": oracle_decision,
            **judge_result,
        }

        rows.append(out)

        pd.DataFrame(rows).to_csv(output_path, index=False)

        print(
            f"Judge decision: {judge_result['judge_decision']} "
            f"| confidence: {judge_result['judge_confidence']:.2f} "
            f"| oracle: {oracle_decision}"
        )

        time.sleep(0.2)

    print("\nDone.")
    print(f"Saved judge results to: {output_path}")


if __name__ == "__main__":
    main()