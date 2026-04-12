import pandas as pd

df = pd.read_csv("data/StudentData.csv")

print("Missing Values:")
print(df.isnull().sum())

# Drop missing values
df = df.dropna()

# Convert categorical → numeric
df = pd.get_dummies(df)

# Save cleaned data
df.to_csv("processed/cleaned_data.csv", index=False)

print("Data preprocessing completed")
