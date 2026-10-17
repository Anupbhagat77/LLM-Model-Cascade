import time
import re
from typing import Dict, Any

from dotenv import load_dotenv
from litellm import completion, completion_cost

from config import TEMPERATURE, USD_TO_EUR_RATE


load_dotenv()


def extract_retry_seconds(error_message: str, default_seconds: int = 35) -> int:
    """
    Extracts retry delay from API error messages.
    If no retry delay is found, returns default_seconds.
    """

    patterns = [
        r"retry in ([0-9.]+)s",
        r'"retryDelay":\s*"([0-9.]+)s"',
        r"retryDelay.*?([0-9.]+)s"
    ]

    for pattern in patterns:
        match = re.search(pattern, error_message, re.IGNORECASE)
        if match:
            try:
                return int(float(match.group(1))) + 5
            except ValueError:
                return default_seconds

    return default_seconds


def is_free_tier_quota_error(error_message: str) -> bool:
    """
    Detects Gemini free-tier quota exhaustion.
    This should not be retried repeatedly because it usually means
    the API key is still connected to a free-tier project or the daily
    free quota is exhausted.
    """

    error_lower = error_message.lower()

    free_tier_indicators = [
        "generate_content_free_tier_requests",
        "generaterequestsperdayperprojectpermodel-freetier",
        "free-tier",
        "free tier"
    ]

    return any(indicator in error_lower for indicator in free_tier_indicators)


def is_retryable_error(error_message: str) -> bool:
    """
    Checks whether an error is retryable.
    We retry temporary rate limits and service errors.
    """

    error_lower = error_message.lower()

    retryable_keywords = [
        "429",
        "rate limit",
        "ratelimit",
        "resource_exhausted",
        "temporarily unavailable",
        "timeout",
        "503",
        "500"
    ]

    return any(keyword in error_lower for keyword in retryable_keywords)


def call_model(
    model_name: str,
    prompt: str,
    max_tokens: int,
    retries: int = 3
) -> Dict[str, Any]:
    """
    Calls an LLM using LiteLLM.

    Returns:
    - answer
    - token usage
    - estimated cost in USD
    - estimated cost in EUR
    - latency
    - success status
    - error message if failed
    """

    last_error = None

    for attempt in range(1, retries + 1):
        start_time = time.time()

        try:
            response = completion(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=TEMPERATURE,
                max_tokens=max_tokens
            )

            end_time = time.time()
            latency_seconds = round(end_time - start_time, 3)

            answer = response["choices"][0]["message"]["content"]

            usage = response.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

            try:
                estimated_cost_usd = completion_cost(completion_response=response)
            except Exception:
                estimated_cost_usd = 0.0

            estimated_cost_eur = estimated_cost_usd * USD_TO_EUR_RATE

            return {
                "model": model_name,
                "answer": answer,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost_usd,
                "estimated_cost_eur": estimated_cost_eur,
                "latency_seconds": latency_seconds,
                "success": True,
                "error": None
            }

        except Exception as e:
            end_time = time.time()
            latency_seconds = round(end_time - start_time, 3)

            error_message = str(e)
            last_error = error_message

            print("\nModel call failed.")
            print(f"Model: {model_name}")
            print(f"Attempt: {attempt}/{retries}")
            print(f"Error: {error_message[:700]}")

            if is_free_tier_quota_error(error_message):
                print("\nFree-tier Gemini quota error detected.")
                print("This means the API key may still be linked to a free-tier project.")
                print("Stopping retries for this request to avoid wasting time.")
                break

            if attempt < retries and is_retryable_error(error_message):
                wait_seconds = extract_retry_seconds(error_message)

                print(
                    f"Retryable error detected. "
                    f"Waiting {wait_seconds} seconds before retrying..."
                )

                time.sleep(wait_seconds)
                continue

            break

    return {
        "model": model_name,
        "answer": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "estimated_cost_eur": 0.0,
        "latency_seconds": 0,
        "success": False,
        "error": last_error
    }