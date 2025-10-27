import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Script started...")

# 1. Load Data
try:
    data = pd.read_csv('heart.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Make sure it's in the same directory.")
    exit()

# 2. Simple Preprocessing
# The dataset is mostly clean. 'target' is our goal (1 = disease, 0 = no disease)
# Let's define our features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']

# 3. Split Data (for testing our model)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split complete.")

# 4. Scale Data
# It's crucial to scale data for many ML models.
# We will fit the scaler on the training data and save it.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# 6. Evaluate Model (Just for us to see)
y_pred = model.predict(X_test_scaled)
print(f"Model Accuracy on Test Set: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 7. --- IMPORTANT: FINAL PRODUCTION MODEL ---
# Now, we train the scaler and model on ALL data for the final app.
# This ensures our app uses the most information possible.

# 7a. Fit the Scaler on ALL X data
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)
print("Final scaler prepared.")

# 7b. Fit the Model on ALL X and y data
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_scaled, y)
print("Final model trained on all data.")

# 8. Save the Model and Scaler
# These .pkl files are what our app will use
joblib.dump(final_model, 'model.pkl')
joblib.dump(final_scaler, 'scaler.pkl')

print("---")
print("Model and scaler saved as 'model.pkl' and 'scaler.pkl'.")
print("Model training script finished.")