import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

models = client.models.list()

print("\nAvailable OpenAI models:\n")
for model in models.data:
    print(model.id)