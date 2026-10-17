from src.prompts import create_mmlu_prompt, create_gsm8k_prompt
from src.evaluator import evaluate_mmlu, evaluate_gsm8k


print("\nTesting MMLU prompt:")
mmlu_prompt = create_mmlu_prompt(
    question="What is the capital of France?",
    choices=["Berlin", "Madrid", "Paris", "Rome"]
)
print(mmlu_prompt)


print("\nTesting GSM8K prompt:")
gsm8k_prompt = create_gsm8k_prompt(
    question="Tom has 3 apples and buys 4 more. How many apples does he have?"
)
print(gsm8k_prompt)


print("\nTesting MMLU evaluator:")
mmlu_eval = evaluate_mmlu(
    model_output="Final answer: C",
    ground_truth="C"
)
print(mmlu_eval)


print("\nTesting GSM8K evaluator:")
gsm8k_eval = evaluate_gsm8k(
    model_output="Final answer: 7",
    ground_truth="#### 7"
)
print(gsm8k_eval)


print("\nPrompt and evaluator test completed.")