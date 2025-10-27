import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

# --- 1. Load Model, Scaler, and Column List ---
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)
    
    print("Model, scaler, and columns loaded successfully.")
except FileNotFoundError:
    st.error("Error: Model files not found. Did you re-run `train_model.py` to create 'model_columns.json'?")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading files: {e}")
    st.stop()


# --- 2. App Title and Description ---
st.set_page_config(page_title="CardioScribe", layout="centered")
st.title("❤️ CardioScribe: Heart Disease Predictor")

# --- ADD THIS WARNING ---
st.warning(
    "**DISCLAIMER:** This application is for educational and portfolio purposes only. "
    "The predictions are based on a model trained on a public dataset and are **not** a substitute for "
    "professional medical advice, diagnosis, or treatment. "
    "Please consult a qualified healthcare provider with any health concerns.",
    icon="⚠️"
)
# --- END WARNING ---

st.write(
    "This app predicts your 10-year risk of heart disease based on medical data. "
    "Fill in the parameters in the sidebar to get your prediction."
)


# --- 3. User Input Sidebar ---
st.sidebar.header("Patient Data")

# Helper function to create inputs
def user_input_features():
    
    # --- Numerical Inputs (Based on your dataset) ---
    age = st.sidebar.slider("Age", 29, 77, 54)
    trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 94, 200, 131)
    chol = st.sidebar.slider("Serum Cholestoral (chol)", 126, 564, 246)
    thalch = st.sidebar.slider("Max Heart Rate Achieved (thalch)", 71, 202, 150)
    oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0)
    ca = st.sidebar.slider("Major Vessels (ca)", 0.0, 4.0, 0.0)
    
    
    # --- Categorical Inputs (Must match the strings in your CSV) ---
    
    # sex (We encoded this manually in train_model.py)
    sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
    
    # cp (Chest Pain Type)
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", 
                             ('asymptomatic', 'atypical angina', 'non-anginal', 'typical angina'))
    
    # fbs (Fasting Blood Sugar) - Error showed 'fbs_True'
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", 
                              (True, False))
    
    # restecg (Resting ECG) - Assuming this was numeric. If it was text, this will need to be a selectbox.
    restecg = st.sidebar.slider("Resting ECG Results (restecg)", 0, 2, 1)

    # exang (Exercise Angina) - Error showed 'exang_True'
    exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", 
                                (True, False))
    
    # slope (Slope of Peak Exercise ST) - Assuming numeric.
    slope = st.sidebar.slider("Slope of Peak Exercise ST (slope)", 0.0, 2.0, 1.0)

    # thal (Thalassemia) - Assuming numeric.
    thal = st.sidebar.slider("Thalassemia (thal)", 0.0, 3.0, 2.0)


    # --- Create Dictionary from inputs ---
    data = {
        'age': age,
        'sex': 1 if sex == 'Male' else 0,  # Replicate manual encoding
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalch': thalch,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    # Convert dictionary to a DataFrame
    input_df = pd.DataFrame(data, index=[0])
    return input_df

# Get user inputs
input_df_raw = user_input_features()


# --- 4. Prediction Logic ---
if st.sidebar.button("Predict"):
    try:
        # --- 4a. Feature Engineering (CRITICAL STEP) ---
        
        # 1. One-hot encode the categorical features from the input
        input_df_processed = pd.get_dummies(input_df_raw)
        
        # 2. Align columns with the model's "memory"
        # This adds any missing columns (e.g., 'cp_non-anginal') and fills them with 0
        input_df_aligned = input_df_processed.reindex(columns=model_columns, fill_value=0)

        # --- 4b. Scale the data ---
        input_scaled = scaler.transform(input_df_aligned)

        # --- 4c. Make prediction ---
        prediction = model.predict(input_scaled)
        
        # [0][1] gives the probability of class 1 (disease)
        # Your target 'num' might be 0 for no disease and 1, 2, 3, 4 for disease
        # We'll take the probability of NOT being 0
        prediction_proba = 1 - model.predict_proba(input_scaled)[0][0]
        risk_percent = round(prediction_proba * 100, 2)
        
        st.write("---")
        st.header("Prediction Result")

        # --- 5. Display Result ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Risk of Heart Disease", value=f"{risk_percent}%")

        with col2:
            # Check if prediction is non-zero (0 = no disease)
            if prediction[0] > 0: 
                st.error("High Risk")
                st.write("The model predicts a **high probability** of heart disease.")
            else:
                st.success("Low Risk")
                st.write("The model predicts a **low probability** of heart disease.")
        
        st.subheader("What this means:")
        st.write(
            f"Based on your data, the model estimates a **{risk_percent}%** probability of having heart disease. "
            "A **High Risk** result indicates your parameters are similar to patients who had heart disease in the dataset."
        )
        st.warning("**Disclaimer:** This is an AI-powered prediction and not a medical diagnosis. Please consult a healthcare professional for any health concerns.")
        

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("This might be because a numeric input (like 'restecg' or 'slope') was actually text in the CSV. If so, change its input in `app.py` from `st.slider` to `st.selectbox` with the text options.")
else:
    st.info("Fill in your data on the left and click 'Predict' to see your risk.")