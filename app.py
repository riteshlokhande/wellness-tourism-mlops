import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="🌴",
    layout="wide"
)

# Cache and Load Model safely
@st.cache_resource
def load_model():
    model_path = "model/wellness_model.joblib"
    if not os.path.exists(model_path):
        # Fallback path if running from root vs subfolder
        model_path = "wellness_model.joblib"
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

model = load_model()

st.title("🌴 Wellness Tourism Package Purchase Prediction")
st.markdown("Provide customer profile details below to predict the likelihood of purchasing the wellness tourism package.")

if model is None:
    st.error("❌ Model artifact (`wellness_model.joblib`) not found! Please ensure your MLOps pipeline has successfully completed training and synced the model.")
else:
    with st.form("prediction_form"):
        st.subheader("Customer Demographics & Profile")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
            city_tier = st.selectbox("City Tier", [1, 2, 3])
            duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=15)

        with col2:
            occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
            number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)

        with col3:
            product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
            preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        st.subheader("Travel Habits & Financials")
        col4, col5, col6 = st.columns(3)

        with col4:
            own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            number_of_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=20, value=2)

        with col5:
            pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
            number_of_children_visiting = st.number_input("Children Visiting (Under 5)", min_value=0, max_value=5, value=0)

        with col6:
            designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
            monthly_income = st.number_input("Monthly Income (₹)", min_value=10000, max_value=200000, value=25000, step=5000)

        submitted = st.form_submit_button("Predict Purchase Likelihood")

        if submitted:
            # Construct input DataFrame matching exact feature schema
            input_data = pd.DataFrame([{
                'Age': age,
                'TypeofContact': type_of_contact,
                'CityTier': city_tier,
                'DurationOfPitch': duration_of_pitch,
                'Occupation': occupation,
                'Gender': gender,
                'NumberOfPersonVisiting': number_of_person_visiting,
                'NumberOfFollowups': number_of_followups,
                'ProductPitched': product_pitched,
                'PreferredPropertyStar': preferred_property_star,
                'MaritalStatus': marital_status,
                'Passport': passport,
                'OwnCar': own_car,
                'NumberOfTrips': number_of_trips,
                'PitchSatisfactionScore': pitch_satisfaction_score,
                'NumberOfChildrenVisiting': number_of_children_visiting,
                'Designation': designation,
                'MonthlyIncome': monthly_income
            }])

            try:
                # Run Inference
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0][1]

                st.markdown("---")
                st.subheader("Prediction Results")
                
                if prediction == 1:
                    st.success(f"🎉 **High Likelihood!** This customer is likely to purchase the wellness tourism package (Probability: {prediction_proba:.2%}).")
                else:
                    st.info(f"💡 **Low Likelihood.** This customer is unlikely to purchase the wellness tourism package (Probability: {prediction_proba:.2%}).")

            except Exception as e:
                st.error(f"❌ Error during inference execution: {e}")
