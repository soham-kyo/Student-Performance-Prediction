import pandas as pd
import joblib
import matplotlib.pyplot as plt

X_test = pd.read_csv("processed/X_test.csv")
y_test = pd.read_csv("processed/y_test.csv")

model = joblib.load("model/model.pkl")

y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Student Performance Prediction")
plt.show()
