import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("analysis_out/settlement_company_totals_compare.csv")

plt.scatter(df["input_count"], df["generated_count"], alpha=0.3)

plt.xlabel("Input companies")
plt.ylabel("Generated companies")

plt.title("Settlement scaling check")

plt.show()