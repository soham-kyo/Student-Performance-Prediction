import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

X_test = pd.read_csv("processed/X_test.csv")
y_test = pd.read_csv("processed/y_test.csv")

model = joblib.load("model/model.pkl")

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
