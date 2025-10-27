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

# --- START NEW FIX SECTION ---

# 2. Handle 'sex' column (which we know is 'Male'/'Female')
data['sex'] = data['sex'].apply(lambda x: 1 if x == 'Male' else 0)
print("Converted 'sex' column.")

# 3. Define Target (y) and drop non-feature columns
# We do this BEFORE encoding to make sure we don't encode 'id' or 'dataset'
y = data['num']
X = data.drop(['id', 'dataset', 'num'], axis=1)
print("Target 'y' and features 'X' separated.")

# 4. Find ALL remaining text columns in X and convert them
# This will find 'cp' (with 'asymptomatic') and any others.
text_cols = X.select_dtypes(include=['object']).columns

if len(text_cols) > 0:
    print(f"Converting categorical text columns: {list(text_cols)}")
    # Use pd.get_dummies to one-hot encode all text columns at once
    # This creates new columns (e.g., cp_asymptomatic) with 0s and 1s
    X = pd.get_dummies(X, columns=text_cols, drop_first=True)
    print("Text columns converted.")
else:
    print("No categorical text columns found to convert.")

# --- END NEW FIX SECTION ---


# 5. Split Data (This was your original line 28)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split complete.")

# 6. Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data scaling complete.")

# 7. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# 8. Evaluate Model (Just for us to see)
y_pred = model.predict(X_test_scaled)
print(f"Model Accuracy on Test Set: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 9. --- IMPORTANT: FINAL PRODUCTION MODEL ---
print("Training final model on all data...")
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X) # X is the final, encoded DataFrame
print("Final scaler prepared.")

final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_scaled, y)
print("Final model trained on all data.")


# 10. Save the Model, Scaler, AND Column List
joblib.dump(final_model, 'model.pkl')
joblib.dump(final_scaler, 'scaler.pkl')

# --- ADD THIS NEW CODE ---
import json
model_columns = list(X.columns)
with open('model_columns.json', 'w') as f:
    json.dump(model_columns, f)
# --- END NEW CODE ---


print("---")
print("Model, scaler, and column list saved.")
print("Model training script finished.")