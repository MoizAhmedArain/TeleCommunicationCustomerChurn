import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# -----------------------------
# Load your trained model + encoder
# -----------------------------
# Assume you already trained and saved your model:
# joblib.dump(rf, "rf_model.pkl")
rf = joblib.load("rf_model.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Customer Churn Prediction")

st.write("Select customer details below to predict churn:")

# Dropdowns / Radios for categorical features
contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
offer = st.selectbox("Offer", ["No Offer", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])
payment = st.radio("Payment Method", ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"])
internet = st.radio("Internet Service", ["DSL", "Fiber optic", "No"])

# Numeric inputs with sliders
tenure = st.slider("Tenure in Months", min_value=0, max_value=72, value=12)
monthly_charge = st.number_input("Monthly Charge", min_value=0.0, max_value=200.0, value=50.0)
profit = st.number_input("Profit", min_value=-500.0, max_value=500.0, value=100.0)

# -----------------------------
# Prepare input for model
# -----------------------------
input_dict = {
    "Contract": contract,
    "Offer": offer,
    "Payment Method": payment,
    "Internet Service": internet,
    "Tenure in Months": tenure,
    "Monthly Charge": monthly_charge,
    "Profit": profit
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Apply same dummy encoding as training
input_encoded = pd.get_dummies(input_df,
                               columns=["Contract","Offer","Payment Method","Internet Service"],
                               drop_first=True)

# Align columns with training data
# (important: ensures same feature order)
model_features = rf.feature_names_in_
for col in model_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[model_features]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):
    pred = rf.predict(input_encoded)[0]
    prob = rf.predict_proba(input_encoded)[0][1]

    if pred == 1:
        st.error(f"⚠️ Customer likely to churn (probability {prob:.2f})")
    else:
        st.success(f"✅ Customer likely to stay (probability {1-prob:.2f})")
