
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = BASE_DIR / "model" / "model.pkl"
COLUMNS_PATH = BASE_DIR / "processed" / "X_train.csv"
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    columns = pd.read_csv(COLUMNS_PATH, nrows=0).columns.tolist()
    return model, columns
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827 0%, #182338 100%);
}
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(120deg, #23395d, #304e78);
    border: 1px solid #49658a;
    margin-bottom: 25px;
}
.hero h1 { color: white; margin-bottom: 8px; }
.hero p { color: #d8e6ff; font-size: 17px; }
div[data-testid="stForm"] {
    border: 1px solid #34445c;
    border-radius: 16px;
    padding: 22px;
    background: rgba(255,255,255,0.025);
}
.result {
    background: #152e2b;
    border: 1px solid #27876f;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-top: 25px;
}
.result-label { color: #a7e8d2; font-size: 17px; }
.result-score { color: #ffffff; font-size: 48px; font-weight: 800; }
.small-note { color: #aab9ce; font-size: 13px; }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero">
    <h1>🎓 Student Performance Prediction</h1>
    <p>Predict an estimated exam score using student study habits,
    academic history, and learning environment.</p>
</div>
""", unsafe_allow_html=True)

# Check required files
if not MODEL_PATH.exists() or not COLUMNS_PATH.exists():
    st.error(
        "Model files are missing. Please train the model first "
        "and ensure model/model.pkl and processed/X_train.csv exist."
    )
    st.stop()

model, model_columns = load_model()

st.markdown("## 📝 Student Details")
st.caption("Enter the details below, then click Predict Exam Score.")
with st.form("prediction_form"):

    st.markdown("### 📚 Academic Habits")

    c1, c2, c3 = st.columns(3)

    with c1:
        hours = st.number_input(
            "Hours studied per week", min_value=0.0,
            max_value=168.0, value=10.0, step=1.0
        )
        attendance = st.number_input(
            "Attendance (%)", min_value=0.0,
            max_value=100.0, value=75.0, step=1.0
        )

    with c2:
        sleep = st.number_input(
            "Sleep hours per day", min_value=0.0,
            max_value=24.0, value=6.0, step=0.5
        )
        previous = st.number_input(
            "Previous score (out of 100)", min_value=0.0,
            max_value=100.0, value=85.0, step=1.0
        )

    with c3:
        tutoring = st.number_input(
            "Tutoring sessions per month", min_value=0,
            max_value=100, value=15
        )
        activity = st.number_input(
            "Physical activity (hours/week)", min_value=0.0,
            max_value=168.0, value=10.0, step=1.0
        )

    st.markdown("### 🏫 Learning Environment")

    c1, c2, c3 = st.columns(3)

    with c1:
        parent = st.selectbox(
            "Parental involvement", ["Low", "Medium", "High"]
        )
        resource = st.selectbox(
            "Access to resources", ["Low", "Medium", "High"]
        )
        extra = st.selectbox(
            "Extracurricular activities", ["Yes", "No"]
        )
        motivation = st.selectbox(
            "Motivation level", ["Low", "Medium", "High"]
        )

    with c2:
        internet = st.selectbox("Internet access", ["Yes", "No"])
        income = st.selectbox(
            "Family income", ["Low", "Medium", "High"]
        )
        teacher = st.selectbox(
            "Teacher quality", ["Low", "Medium", "High"]
        )
        school = st.selectbox("School type", ["Public", "Private"])

    with c3:
        peer = st.selectbox(
            "Peer influence", ["Negative", "Neutral", "Positive"]
        )
        disability = st.selectbox(
            "Learning disability", ["No", "Yes"]
        )
        education = st.selectbox(
            "Parental education",
            ["High School", "College", "Postgraduate"]
        )
        distance = st.selectbox(
            "Distance from home", ["Near", "Moderate", "Far"]
        )

    st.markdown("### 👤 Additional Details")
    c1, c2 = st.columns(2)

    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])

    with c2:
        st.markdown(
            '<p class="small-note">Your entries are used to generate '
            'a prediction from the locally saved model.</p>',
            unsafe_allow_html=True
        )

    submitted = st.form_submit_button(
        "🎯 Predict Exam Score", use_container_width=True
    )
if submitted:
    input_data = pd.DataFrame([{
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
    encoded_data = pd.get_dummies(input_data)
    encoded_data = encoded_data.reindex(
        columns=model_columns, fill_value=0
    )

    prediction = float(np.asarray(model.predict(encoded_data)).reshape(-1)[0])
    prediction = float(np.clip(prediction, 0, 100))

    st.markdown(f"""
    <div class="result">
        <div class="result-label">🎯 Predicted Exam Score</div>
        <div class="result-score">{prediction:.2f} / 100</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(prediction / 100)

    if prediction >= 80:
        st.success("Excellent predicted performance! Keep it up.")
    elif prediction >= 60:
        st.info("Good progress. Consistent preparation can help improve performance.")
    elif prediction >= 40:
        st.warning("There is room to improve. Focus on study habits and revision.")
    else:
        st.warning("Consider additional academic support and a structured study plan.")

    st.caption(
        "This is a model estimate, not a guaranteed exam result. "
        "Prediction quality depends on the training data and model."
    )
