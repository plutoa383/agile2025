# trainUberML_GB.py
# Uber ride success prediction using GradientBoostingClassifier
# Saves trained model, test predictions, and confusion matrix
# Dataset: ./dataset/uber_data.csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# --------------------------
# Load dataset
# --------------------------
csv_path = "dataset/uber_data_cleaned.csv"
df = pd.read_csv(csv_path)

# --------------------------
# Target variable
# --------------------------
df['Ride_Success'] = df['Booking Status_Completed']

df = df.drop(columns=[
    'Booking Status_Cancelled by Driver',
    'Booking Status_Completed',
    'Booking Status_Incomplete',
    'Booking Status_No Driver Found'
])

# --------------------------
# Features
# --------------------------
X = df.drop(columns=['Ride_Success', 'Booking ID', 'Customer ID', 'Date', 'Time'])
y = df['Ride_Success']

categorical_cols = ['Pickup Location', 'Drop Location']
X = pd.get_dummies(X, columns=categorical_cols)

# --------------------------
# Train/test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# Scale features
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# Train Gradient Boosting model
# --------------------------
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# --------------------------
# Evaluate model
# --------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# --------------------------
# Save test predictions with confidence
# --------------------------
test_predictions = pd.DataFrame({
    'Ride_Success_Actual': y_test,
    'Ride_Success_Predicted': y_pred,
    'Success_Probability': y_prob
})
test_predictions.to_csv("test_predictions.csv", index=False)

# --------------------------
# Sample prediction
# --------------------------
sample = X_test_scaled[0].reshape(1, -1)
prediction = model.predict(sample)[0]
confidence = model.predict_proba(sample)[0][1]

print("\nSample Prediction:")
print("Prediction:", "Success" if prediction == 1 else "Not Success")
print("Confidence:", round(confidence * 100, 2), "%")

# --------------------------
# Save model, scaler, and features
# --------------------------
model_filename = "modelUberML.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'feature_columns': X.columns.tolist()
    }, file)

print(f"\nTrained model saved to {model_filename}")
print("Confusion matrix saved as confusion_matrix.png")
print("Test predictions saved as test_predictions.csv")
