def create_mmlu_prompt(question: str, choices: list) -> str:
    """
    Creates a strict prompt format for MMLU multiple-choice questions.

    Expected output from the model:
    Final answer: A
    """

    option_letters = ["A", "B", "C", "D"]

    formatted_options = []
    for letter, choice in zip(option_letters, choices):
        formatted_options.append(f"{letter}) {choice}")

    options_text = "\n".join(formatted_options)

    prompt = f"""
You are answering a multiple-choice benchmark question.

Question:
{question}

Options:
{options_text}

Instructions:
Select the single best answer.
Do not explain your reasoning.
Do not write any extra text.
Return only one line in exactly this format:

Final answer: A
""".strip()

    return prompt


def create_gsm8k_prompt(question: str) -> str:
    """
    Creates a prompt format for GSM8K math word problems.

    The model is allowed to reason briefly, but must end with:
    Final answer: 42
    """

    prompt = f"""
You are solving a math word problem.

Question:
{question}

Instructions:
Solve the problem step by step.
Keep the reasoning concise.
At the end, write the final numeric answer on a separate line in exactly this format:

Final answer: 42
""".strip()

    return prompt