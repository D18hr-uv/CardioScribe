import streamlit as st
import pandas as pd
import joblib

# --- 1. Load Model and Scaler ---
# Load the pre-trained model and scaler
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Please run `train_model.py` first.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading files: {e}")
    st.stop()


# --- 2. App Title and Description ---
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Risk Predictor")
st.write(
    "This app predicts your 10-year risk of heart disease based on medical data. "
    "Fill in the parameters in the sidebar to get your prediction."
)
st.write("---")


# --- 3. User Input Sidebar ---
st.sidebar.header("Patient Data")

# Helper function to create sliders
def user_input_features():
    # Input field labels and their (min, default, max) values
    # Based on the dataset's columns
    params = {
        'age': st.sidebar.slider("Age", 29, 77, 54),
        'sex': st.sidebar.selectbox("Sex", (0, 1), format_func=lambda x: "Female" if x == 0 else "Male"),
        'cp': st.sidebar.slider("Chest Pain Type (cp)", 0, 3, 1),
        'trestbps': st.sidebar.slider("Resting Blood Pressure (trestbps)", 94, 200, 131),
        'chol': st.sidebar.slider("Serum Cholestoral (chol)", 126, 564, 246),
        'fbs': st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", (0, 1), format_func=lambda x: "False" if x == 0 else "True"),
        'restecg': st.sidebar.slider("Resting ECG Results (restecg)", 0, 2, 1),
        'thalach': st.sidebar.slider("Max Heart Rate Achieved (thalach)", 71, 202, 150),
        'exang': st.sidebar.selectbox("Exercise Induced Angina (exang)", (0, 1), format_func=lambda x: "No" if x == 0 else "Yes"),
        'oldpeak': st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0),
        'slope': st.sidebar.slider("Slope of Peak Exercise ST (slope)", 0, 2, 1),
        'ca': st.sidebar.slider("Major Vessels Colored by Flourosopy (ca)", 0, 4, 0),
        'thal': st.sidebar.slider("Thalassemia (thal)", 0, 3, 2),
    }
    
    # Convert dictionary to a DataFrame
    data = pd.DataFrame(params, index=[0])
    return data

# Get user inputs
input_df = user_input_features()


# --- 4. Prediction Logic ---
if st.sidebar.button("Predict"):
    try:
        # Scale the user's input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Get prediction probability
        # [0][1] gives the probability of class 1 (disease)
        prediction_proba = model.predict_proba(input_scaled)[0][1]
        risk_percent = round(prediction_proba * 100, 2)
        
        st.write("---")
        st.header("Prediction Result")

        # --- 5. Display Result ---
        
        # Use columns for a cleaner layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Display the main risk metric
            st.metric(
                label="Risk of Heart Disease",
                value=f"{risk_percent}%"
            )

        with col2:
            # Display the binary prediction
            if prediction[0] == 1:
                st.error("High Risk")
                st.write("The model predicts a **high probability** of heart disease.")
            else:
                st.success("Low Risk")
                st.write("The model predicts a **low probability** of heart disease.")
        
        st.subheader("What this means:")
        st.write(
            f"Based on your data, the model estimates a **{risk_percent}%** probability of having heart disease. "
            "A **High Risk** result (prediction = 1) indicates that your parameters are similar to patients who had heart disease in the dataset. "
            "A **Low Risk** result (prediction = 0) indicates your parameters are similar to those who did not."
        )
        st.warning("**Disclaimer:** This is an AI-powered prediction and not a medical diagnosis. Please consult a healthcare professional for any health concerns.")
        
        # (Optional) Show the data you received
        st.subheader("Your Input Data:")
        st.dataframe(input_df)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
else:
    st.info("Fill in your data on the left and click 'Predict' to see your risk.")