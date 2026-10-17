import pandas as pd

judge = pd.read_csv("results/judge_results_20260628_161248.csv")

print(judge["judge_decision"].value_counts())
print()
print(judge.groupby("judge_decision")["judge_confidence"].describe())