import pandas as pd

df = pd.read_csv("data/StudentData.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())
