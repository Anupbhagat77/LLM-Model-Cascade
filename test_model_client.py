from config import SMALL_MODEL, LARGE_MODEL
from src.model_client import call_model


test_prompt = "Answer only with the correct option letter. What is 2 + 2? A) 3 B) 4 C) 5 D) 6"

print("\nTesting small model through model_client.py...")
small_result = call_model(
    model_name=SMALL_MODEL,
    prompt=test_prompt,
    max_tokens=20
)
print(small_result)

print("\nTesting large model through model_client.py...")
large_result = call_model(
    model_name=LARGE_MODEL,
    prompt=test_prompt,
    max_tokens=20
)
print(large_result)

print("\nmodel_client.py test completed.")