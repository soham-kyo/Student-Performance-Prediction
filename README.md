# 🎓 Student Performance Prediction

> A Machine Learning project that predicts student exam scores based on academic, behavioral, and lifestyle factors.

---

## 📌 Overview

This project uses **Linear Regression** to predict student performance by analyzing factors such as:

- Study hours 📚
- Attendance 🏫
- Sleep patterns 😴
- Previous scores 📊
- Lifestyle & environment 🌱

---

## 🚀 Features

- ✅ Data cleaning & preprocessing
- ✅ Train-test split
- ✅ Model training (Linear Regression)
- ✅ Evaluation using MAE & R² Score
- ✅ Data visualization
- ✅ Real-time prediction via user input

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

## 📂 Project Structure

```bash
Student-Performance-Prediction/
│
├── data/                  # Raw dataset
│   └── StudentData.csv
│
├── processed/             # Processed datasets
│   ├── cleaned_data.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── model/                 # Saved model
│   └── model.pkl
│
├── src/                   # Source code
│   ├── step1_load_data.py
│   ├── step2_preprocess.py
│   ├── step3_split.py
│   ├── step4_train.py
│   ├── step5_evaluate.py
│   ├── step6_visualize.py
│   └── step7_predict.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the scripts in order:

```bash
python src/step1_load_data.py
python src/step2_preprocess.py
python src/step3_split.py
python src/step4_train.py
python src/step5_evaluate.py
python src/step6_visualize.py
python src/step7_predict.py
```

---

## 📊 Workflow

| Step | Description             |
| ---- | ----------------------- |
| 1️⃣   | Load dataset            |
| 2️⃣   | Clean & preprocess data |
| 3️⃣   | Split into train/test   |
| 4️⃣   | Train model             |
| 5️⃣   | Evaluate performance    |
| 6️⃣   | Visualize results       |
| 7️⃣   | Predict new values      |

---

## 🧠 Input Guidelines

### 🔢 Numerical Inputs

- `Hours_Studied` → per week
- `Attendance` → % (0–100)
- `Sleep_Hours` → per day
- `Previous_Scores` → out of 100
- `Tutoring_Sessions` → per month
- `Physical_Activity` → hours/week

### 🔤 Categorical Inputs _(use exact values)_

- Low / Medium / High
- Yes / No
- Public / Private
- Negative / Neutral / Positive
- High School / College / Postgraduate
- Near / Moderate / Far
- Male / Female

---

## ⚠️ Important Notes

- Dataset must be inside `/data` folder
- Target column must be: `Exam_Score`
- Run steps in correct order
- Do not rename files
- Input format must match expected values

---

## 📈 Output

- Model Accuracy (R² Score)
- Mean Absolute Error (MAE)
- Graphs (Actual vs Predicted)
- Predicted Exam Score

---

## 🎯 Objective

To demonstrate how Machine Learning can be used to **analyze student behavior and predict academic performance** effectively.

---

## 🔮 Future Improvements

- 🌐 Deploy as a web app (Streamlit/Flask)
- 📊 Add more advanced models
- ⚡ Improve accuracy
- 🎨 Better UI for predictions

---

## 👨‍💻 Author

**Soham Patil**

---

⭐ _If you found this project useful, consider giving it a star!_
