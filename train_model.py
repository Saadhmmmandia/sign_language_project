import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
data = pd.read_csv("dataset/hand_sign_data.csv")

print("Original dataset shape:", data.shape)

# CLEAN DATA (VERY IMPORTANT)
data = data.dropna()
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

print("Clean dataset shape:", data.shape)

# Split features and labels
X = data.drop("label", axis=1)
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Model
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

# Train
model.fit(X_train, y_train)

print("Model training completed!")

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy * 100, "%")

# Detailed report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(model, "models/sign_model.pkl")

print("\nModel saved at models/sign_model.pkl")