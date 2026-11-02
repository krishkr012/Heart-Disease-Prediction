import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="",
    layout="wide"
)

# ---------------- LOAD FILES ----------------
model = joblib.load("knn_Heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#e63946;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:#555;
    margin-bottom:30px;
}

.card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}

.stButton > button {
    width:100%;
    height:55px;
    border:none;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    color:white;
    background:linear-gradient(90deg,#ff416c,#ff4b2b);
}

.stButton > button:hover {
    background:linear-gradient(90deg,#ff4b2b,#ff416c);
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Health Dashboard")

    st.info("""
    This system predicts the risk of heart attack
    using Machine Learning algorithms.

    """)

    st.markdown("---")

    st.subheader("Tips for Healthy Heart")

    st.write("✅ Exercise Regularly")
    st.write("✅ Maintain Healthy Diet")
    st.write("✅ Control Cholesterol")
    st.write("✅ Monitor Blood Pressure")
    st.write("✅ Avoid Smoking")

# ---------------- HEADER ----------------
st.markdown(
    "<div class='title'>Heart Attack Risk Prediction System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Machine Learning Based Health Assessment Dashboard</div>",
    unsafe_allow_html=True
)

# ---------------- INPUT FORM ----------------
st.markdown("### Enter Patient Information")

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 18, 100, 40)

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        80, 200, 120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        100, 600, 200
    )

with col2:

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar >120 mg/dL",
        [0, 1]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60, 220, 150
    )

    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0, 6.0, 1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

st.markdown("---")

# ---------------- PREDICT BUTTON ----------------
if st.button("🔍 Predict Heart Risk"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    probability = model.predict_proba(scaled_input)[0][1]

    st.markdown("---")

    st.subheader("Prediction Result")

    st.progress(int(probability * 100))

    colA, colB = st.columns(2)

    with colA:
        st.metric(
            "Risk Score",
            f"{probability*100:.1f}%"
        )

    with colB:
        if prediction == 1:
            st.error("⚠️ High Risk Detected")
        else:
            st.success("✅ Low Risk Detected")

    st.markdown("---")

    risk_factors = []

    if cholesterol > 240:
        risk_factors.append("High Cholesterol")

    if resting_bp > 140:
        risk_factors.append("High Blood Pressure")

    if age > 55:
        risk_factors.append("Older Age")

    if risk_factors:

        st.subheader("⚠️ Detected Risk Factors")

        for factor in risk_factors:
            st.warning(factor)

    if prediction == 1:

        st.subheader("💡 Health Recommendations")

        st.info("""
        • Consult a Cardiologist

        • Reduce Cholesterol Intake

        • Exercise Regularly

        • Monitor Blood Pressure

        • Follow Healthy Lifestyle
        """)

    else:

        st.success("""
        Great! Your current health indicators
        show a relatively lower risk profile.
        Continue maintaining a healthy lifestyle.
        """)