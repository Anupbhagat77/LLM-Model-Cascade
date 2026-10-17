<div align="center">

# 🧠 LLM Model Cascade for Cost-Efficient Inference

**M.Sc. thesis implementation (HTW Berlin): can a cheap LLM answer most questions, with a strong LLM stepping in only when needed, and stay just as accurate?**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LiteLLM](https://img.shields.io/badge/LiteLLM-111111?style=flat-square)
![OpenAI](https://img.shields.io/badge/GPT--4.1-412991?style=flat-square&logo=openai&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini%202.5%20Flash%20Lite-4285F4?style=flat-square&logo=googlegemini&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HF%20Datasets-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)

</div>

---

## 📌 The question

Frontier LLMs are accurate but expensive. A **model cascade** sends every query to a small, cheap model first and escalates to a large model only when the small model's answer can't be trusted. This project measures the **cost, quality and latency trade-off** of that idea on two standard benchmarks.

## ⚙️ Setup

| | |
|---|---|
| **Small model** | Gemini 2.5 Flash Lite |
| **Large model** | GPT-4.1 |
| **Judge model** | GPT-4o-mini (for the realistic router) |
| **Benchmarks** | 50 × **MMLU** (multiple choice) + 50 × **GSM8K** (math word problems), same seeded sample for every scenario |
| **Metrics** | Accuracy, quality retention vs. GPT-4.1, tokens, cost (USD/EUR), latency, escalation rate, router errors |
| **Success target** | Quality retention ≥ 95% of the GPT-4.1 baseline |

## 🔀 Scenarios compared

1. **GPT-4.1 only:** the quality and cost baseline
2. **Gemini only:** the cheapest option
3. **Oracle cascade:** escalate only when the small model is *actually* wrong. It uses ground truth, so it is the **theoretical upper bound** for any router.
4. **LLM-as-judge cascade:** a realistic router. GPT-4o-mini checks the small model's answer *without* seeing the ground truth and returns `ACCEPT` / `ESCALATE` with a confidence score.

## 📊 Results (100 questions)

| Scenario | Accuracy | Quality retention | Cost vs. GPT-4.1 | Escalation | Avg. latency |
|---|---|---|---|---|---|
| GPT-4.1 only | 85% | 100% | baseline | n/a | 1.14 s |
| Gemini only | 76% | 89.4% | **−93.9%** | n/a | 0.66 s |
| Oracle cascade (upper bound) | **90%** | **105.9%** | **−59.8%** | 24% | 1.03 s |
| **LLM-as-judge cascade** | 84% | **98.8%** ✅ | **−25.7%** | 64% | 2.89 s |

### Key findings

- 💡 **The upper bound is big.** With perfect routing, the cascade is *cheaper and more accurate* than GPT-4.1 alone, cutting cost by 59.8% at 105.9% quality retention. On the oracle run, GSM8K reached 90% vs. 82% for GPT-4.1.
- ✅ **A realistic router still pays off.** The LLM-as-judge cascade kept **98.8% of GPT-4.1's quality** (above the 95% target) while cutting cost by **25.7%**.
- ⚖️ **The judge is cautious.** It escalated 64% of questions: only **3% false accepts**, but **43% unnecessary escalations**. That caution protects quality and is exactly where the remaining savings are lost. Better routers close the gap to the oracle.
- 🔢 **Tokens ≠ cost.** The oracle cascade used **42.6% more tokens** than GPT-4.1 but cost 59.8% less, because the cheap model's tokens are far cheaper.
- ⏱️ **Latency trade-off.** Accepted questions averaged 0.62 s vs. 2.35 s for escalated ones, and the judge step adds overhead.

All raw per-question results and summaries are in [`results/`](results/).

## 🗂️ Project structure

```
config.py                  # models, token limits, quality threshold, seed, USD→EUR rate
src/
├── benchmark_loader.py    # seeded MMLU + GSM8K samples (Hugging Face datasets)
├── prompts.py             # strict prompt formats ("Final answer: …")
├── model_client.py        # LiteLLM calls with retries, token & cost tracking
└── evaluator.py           # answer extraction + exact-match scoring
run_baseline.py            # Scenario: GPT-4.1 only
run_cascade.py             # Scenario: oracle cascade (small → large)
run_judge_cascade.py       # Scenario: LLM-as-judge router (GPT-4o-mini)
analyze_results.py         # baseline vs. cascade comparison
analyze_routing.py         # routing / escalation breakdown
analyze_judge_results.py   # judge router accuracy, false accept / escalate
test_*.py, smoke_test.py   # quick checks for loader, client, prompts
results/                   # final experiment outputs (CSV / TXT)
```

## 🚀 Reproduce

```bash
git clone https://github.com/Anupbhagat77/<repo-name>.git
cd <repo-name>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your OpenAI + Gemini API keys

python smoke_test.py        # check API access
python run_baseline.py      # GPT-4.1 only
python run_cascade.py       # oracle cascade
python run_judge_cascade.py # LLM-as-judge cascade
python analyze_results.py && python analyze_routing.py && python analyze_judge_results.py
```

> ⚠️ Running the experiments calls paid APIs. The full 100-question run cost well under €0.20 in total.

## ⚠️ Limitations

- 100 questions in total, enough to show the trade-off but not for fine-grained statistical claims.
- The oracle cascade uses ground truth for routing, so it is an upper bound, not a deployable system.
- Costs are estimates from LiteLLM's pricing tables at the time of the experiments.

## 🔮 Future work

- Cheaper, better-calibrated routers (self-consistency, log-prob confidence, trained classifiers)
- More benchmarks and larger samples
- Multi-tier cascades with more than two models

---

<div align="center">
<sub>M.Sc. Project Management & Data Science, HTW Berlin · by <b>Anup Bhagat</b> · <a href="https://anupbhagat77.github.io">Portfolio</a> · <a href="https://www.linkedin.com/in/-anup-bhagat-">LinkedIn</a></sub>
</div>
