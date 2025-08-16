import streamlit as st
import joblib
import pandas as pd

# Load pre-trained model and encoders
clf = joblib.load('model.pkl')
encoders = joblib.load('encoders.pkl')
target_le = joblib.load('target_le.pkl')

st.title("Student Final Grade Prediction")

# Input widgets
age = st.number_input('Age', min_value=12, max_value=20, step=1)
gender = st.selectbox('Gender', ['M', 'F'])
study_hours = st.number_input('Study Hours', min_value=0, max_value=100, step=1)
attendance = st.number_input('Attendance (%)', min_value=0, max_value=100, step=1)
homework_rate = st.slider('Homework Rate (0.0 - 1.0)', 0.0, 1.0, step=0.01)
participation = st.selectbox('Participation', ['High', 'Medium', 'Low'])
favorite_subject = st.selectbox('Favorite Subject', ['Math', 'Science', 'English'])

# Prepare the input dictionary
input_dict = {
    "Age": age,
    "Gender": gender,
    "StudyHours": study_hours,
    "Attendance": attendance,
    "HomeworkRate": homework_rate,
    "Participation": participation,
    "FavoriteSubject": favorite_subject
}

# Encode categorical inputs
input_encoded = []
for feature in ["Age", "Gender", "StudyHours", "Attendance", "HomeworkRate", "Participation", "FavoriteSubject"]:
    val = input_dict[feature]
    if feature in encoders:
        val = encoders[feature].transform([val])[0]
    input_encoded.append(val)

# Button to predict
if st.button("Predict Final Grade"):
    # Create DataFrame with proper feature names for model input
    input_df = pd.DataFrame([input_encoded], columns=["Age", "Gender", "StudyHours", "Attendance", "HomeworkRate", "Participation", "FavoriteSubject"])
    pred = clf.predict(input_df)
    pred_label = target_le.inverse_transform(pred)[0]
    st.success(f"Predicted Final Grade: {pred_label}")
