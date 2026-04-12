import numpy as np
import joblib
import pandas as pd

model = joblib.load("model/model.pkl")

print("\n===== STUDENT PERFORMANCE PREDICTION =====\n")

print("Enter details carefully:\n")

# Numerical inputs
hours = float(input("Hours Studied (per week): "))
attendance = float(input("Attendance (%): "))
sleep = float(input("Sleep Hours (per day): "))
previous = float(input("Previous Score (out of 100): "))
tutoring = float(input("Tutoring Sessions (per month): "))
activity = float(input("Physical Activity (hrs/week): "))

# Categorical inputs
parent = input("Parental Involvement (Low/Medium/High): ")
resource = input("Access to Resources (Low/Medium/High): ")
extra = input("Extracurricular (Yes/No): ")
motivation = input("Motivation (Low/Medium/High): ")
internet = input("Internet (Yes/No): ")
income = input("Family Income (Low/Medium/High): ")
teacher = input("Teacher Quality (Low/Medium/High): ")
school = input("School Type (Public/Private): ")
peer = input("Peer Influence (Negative/Neutral/Positive): ")
disability = input("Learning Disability (Yes/No): ")
education = input("Parental Education (High School/College/Postgraduate): ")
distance = input("Distance (Near/Moderate/Far): ")
gender = input("Gender (Male/Female): ")

# Create dataframe (IMPORTANT: must match training columns)
data = pd.DataFrame([{
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Sleep_Hours": sleep,
    "Previous_Scores": previous,
    "Tutoring_Sessions": tutoring,
    "Physical_Activity": activity,
    "Parental_Involvement": parent,
    "Access_to_Resources": resource,
    "Extracurricular_Activities": extra,
    "Motivation_Level": motivation,
    "Internet_Access": internet,
    "Family_Income": income,
    "Teacher_Quality": teacher,
    "School_Type": school,
    "Peer_Influence": peer,
    "Learning_Disabilities": disability,
    "Parental_Education_Level": education,
    "Distance_from_Home": distance,
    "Gender": gender
}])

# Convert same as training
data = pd.get_dummies(data)

# Align columns
model_columns = pd.read_csv("processed/X_train.csv").columns
data = data.reindex(columns=model_columns, fill_value=0)

# Prediction
prediction = model.predict(data)

print("\n🎯 Predicted Exam Score:", round(float(np.squeeze(prediction)), 2))
