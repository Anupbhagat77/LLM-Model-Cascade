import os
import time
from dotenv import load_dotenv
from litellm import completion

from config import SMALL_MODEL, LARGE_MODEL, TEMPERATURE


load_dotenv()


def call_model(model_name: str, prompt: str):
    start_time = time.time()

    response = completion(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=50
    )

    end_time = time.time()
    latency = end_time - start_time

    answer = response["choices"][0]["message"]["content"]

    usage = response.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    return {
        "model": model_name,
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": round(latency, 3)
    }


if __name__ == "__main__":
    test_prompt = "Answer only with the correct option letter. What is 2 + 2? A) 3 B) 4 C) 5 D) 6"

    print("\nTesting small model...")
    small_result = call_model(SMALL_MODEL, test_prompt)
    print(small_result)

    print("\nTesting large model...")
    large_result = call_model(LARGE_MODEL, test_prompt)
    print(large_result)

    print("\nSmoke test completed.")