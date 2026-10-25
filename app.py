import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="🌴",
    layout="wide"
)

@st.cache_resource
def load_model():
    """
    Loads the serialized end-to-end Scikit-Learn composite pipeline.
    """
    model_path = 'model/wellness_model.joblib'
    return joblib.load(model_path)

model = load_model()

st.title("🌴 Wellness Tourism Package Prediction Portal")
st.markdown("""
This application utilizes our production-tuned **XGBoost & Scikit-Learn Composite Pipeline** to predict whether a customer is likely to purchase the Wellness Tourism Package (`ProdTaken`). 
Configure the customer profile parameters in the sidebar and click **Predict Purchase Probability** to evaluate conversion potential.
""")

# Sidebar inputs for customer profile
st.sidebar.header("Customer Profile Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 60, 35)
    type_of_contact = st.sidebar.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.sidebar.slider("Duration of Pitch (minutes)", 5, 35, 15)
    occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    number_of_person_visiting = st.sidebar.slider("Number of Persons Visiting", 1, 5, 2)
    number_of_followups = st.sidebar.slider("Number of Follow-ups", 1, 6, 3)
    product_pitched = st.sidebar.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    preferred_property_star = st.sidebar.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Unmarried", "Divorced"])
    number_of_trips = st.sidebar.slider("Number of Trips per Year", 1, 8, 2)
    passport = st.sidebar.selectbox("Has Passport?", [0, 1])
    pitch_satisfaction_score = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.sidebar.selectbox("Owns a Car?", [0, 1])
    number_of_children_visiting = st.sidebar.slider("Number of Children Visiting (under 5)", 0, 3, 0)
    designation = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.sidebar.number_input("Monthly Income (INR)", 10000, 100000, 25000, step=5000)

    data = {
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
        'NumberOfTrips': number_of_trips,
        'Passport': passport,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'OwnCar': own_car,
        'NumberOfChildrenVisiting': number_of_children_visiting,
        'Designation': designation,
        'MonthlyIncome': monthly_income
    }
    
    return pd.DataFrame([data])

input_df = user_input_features()

# Main panel display
st.subheader("Customer Profile Summary")
st.dataframe(input_df)

if st.button("Predict Purchase Probability", type="primary"):
    # Perform prediction using the composite pipeline
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]
    
    st.markdown("---")
    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.success(f"✅ **High Conversion Potential:** This customer is **likely** to purchase the Wellness Tourism Package (Conversion Probability: `{prediction_proba:.2%}`).")
    else:
        st.warning(f"⚠️ **Low Conversion Potential:** This customer is **unlikely** to purchase the Wellness Tourism Package (Conversion Probability: `{prediction_proba:.2%}`).")
