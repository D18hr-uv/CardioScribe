## ❤️ CardioScribe: Interactive Heart Disease Predictor


#### Link: https://cardioscribe.streamlit.app/

An end-to-end machine learning application that predicts the 10-year risk of heart disease based on user-inputted medical data.

### ⚠️ Medical Disclaimer

This application is for educational and portfolio purposes only. The predictions are based on a model trained on a public dataset and are not a substitute for professional medical advice, diagnosis, or treatment. Please consult a qualified healthcare provider with any health concerns.

### 📋 Project Overview

This project demonstrates the complete end-to-end lifecycle of a data science project:

Data Cleaning & Preprocessing: A raw, complex dataset (heart.csv) containing null values, text-based categorical features, and multiple data types was thoroughly cleaned and prepared for modeling.

Model Training: A RandomForestClassifier was trained on the processed data. The model, scaler, and feature list were serialized and saved using joblib and json.

Interactive Web App: A Streamlit application (app.py) provides an intuitive user interface that:

Accepts 13 key medical inputs from the user.

Performs the exact same data processing (one-hot encoding, column alignment) on the user's input.

Loads the pre-trained model and scaler to generate a real-time risk prediction.

Presents the result as a clear percentage and "High Risk" / "Low Risk" classification.

### ✨ Features

Interactive Sidebar: All 13 medical parameters are collected via sliders and select boxes.

Real-time Prediction: Get an instant risk score and probability by clicking "Predict".

Clear Results: Risk is presented as a percentage and a clear "Low Risk" (Green) or "High Risk" (Red) message.

Built-in Disclaimer: Ensures responsible use of the application.

### 💻 Tech Stack

Python: The core programming language.

Pandas: For data loading, cleaning, and manipulation.

Scikit-learn: For model training (RandomForestClassifier), data scaling (StandardScaler), and preprocessing.

Streamlit: For building and serving the interactive web application.

Joblib & JSON: For saving and loading the trained model, scaler, and feature columns.

### 🚀 How to Run Locally

Follow these steps to run the application on your own machine.

1. Prerequisites

Python 3.8 or newer

pip (Python package installer)

2. Clone the Repository

git clone [https://github.com/your-username/your-CardioScribe-repo.git](https://github.com/your-username/your-CardioScribe-repo.git)
cd your-CardioScribe-repo


3. Install Dependencies

Install all the required libraries from the requirements.txt file.

pip install -r requirements.txt


(If you haven't created a requirements.txt file yet, create one and add streamlit, pandas, scikit-learn, and numpy to it.)

4. Run the Model Training Script (Optional)

If you want to re-train the model from scratch, run the training script. This will regenerate model.pkl, scaler.pkl, and model_columns.json.

python train_model.py


5. Run the Streamlit App

This is the main command to start the web application.

streamlit run app.py


Your default web browser will automatically open, and you can start using the CardioScribe app!