# Code Cell: Fully Written app.py File via Cell Magic
import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(
    page_title="Wellness Tourism Predictor",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Wellness Tourism Package Predictor")
st.write("Production MLOps Dashboard powered by Scikit-Learn, XGBoost, and MLflow.")

@st.cache_resource
def load_production_model():
    model_path = 'model/wellness_model.joblib'
    if not os.path.exists(model_path):
        return None, "Model file not found in 'model/wellness_model.joblib'. Please verify file structure."
    try:
        model = joblib.load(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

model, load_error = load_production_model()

if model is None:
    st.error("⚠️ **Model Loading Failed due to Environment/Serialization Mismatch**")
    st.markdown(f"**Error Details:** `{load_error}`")
else:
    st.success("✅ Production model binary (`wellness_model.joblib`) loaded successfully!")

    st.sidebar.header("Customer Demographic & Pitch Inputs")
    age = st.sidebar.slider("Age", 18, 70, 35)
    type_of_contact = st.sidebar.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
    occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    number_of_person = st.sidebar.slider("Persons Visiting", 1, 5, 2)
    pref_property_star = st.sidebar.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    trips_per_year = st.sidebar.slider("Trips per Year", 1, 8, 3)
    passport = st.sidebar.selectbox("Has Passport?", ["Yes", "No"])
    pitch_satisfaction = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 4)
    own_car = st.sidebar.selectbox("Owns a Car?", ["Yes", "No"])
    children_visiting = st.sidebar.slider("Children Visiting (< 5 yrs)", 0, 3, 1)
    designation = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.sidebar.number_input("Monthly Income (₹)", value=45000, step=5000)
    product_pitch_score = st.sidebar.slider("Product Pitch Score", 1, 5, 3)

    if st.button("Run Conversion Prediction", type="primary"):
        input_data = pd.DataFrame({
            'Age': [age],
            'TypeofContact': [type_of_contact],
            'CityTier': [city_tier],
            'Occupation': [occupation],
            'Gender': [gender],
            'NumberOfPersonVisiting': [number_of_person],
            'PreferredPropertyStar': [pref_property_star],
            'MaritalStatus': [marital_status],
            'NumberOfTrips': [trips_per_year],
            'Passport': [1 if passport == "Yes" else 0],
            'PitchSatisfactionScore': [pitch_satisfaction],
            'OwnCar': [1 if own_car == "Yes" else 0],
            'NumberOfChildrenVisiting': [children_visiting],
            'Designation': [designation],
            'MonthlyIncome': [monthly_income],
            'ProductPitchScore': [product_pitch_score]
        })

        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

            st.divider()
            st.subheader("📊 Inference Results")
            if prediction == 1:
                st.success(f"✅ **High Probability of Conversion (Class 1)**")
                if probability is not None:
                    st.metric(label="Predicted Conversion Confidence", value=f"{probability * 100:.2f}%")
                st.balloons()
            else:
                st.info(f"ℹ️ **Low Probability of Conversion (Class 0)**")
                if probability is not None:
                    st.metric(label="Predicted Conversion Confidence", value=f"{(1 - probability) * 100:.2f}% (Non-Buyer)")
        except Exception as pred_err:
            st.error(f"❌ Error during inference execution: {pred_err}")
