import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

X_train = pd.read_csv("processed/X_train.csv")
y_train = pd.read_csv("processed/y_train.csv")

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "model/model.pkl")

print("Model trained and saved")
