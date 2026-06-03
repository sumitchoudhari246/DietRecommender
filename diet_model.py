# diet_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("user_health_dataset.csv")

# Drop unwanted columns
data = data.drop(columns=["Physical_Activity_Level", "Daily_Caloric_Intake"], errors='ignore')

# Remove rows with Obesity
data = data[data["Disease_Type"].str.lower() != "obesity"]

# Recalculate BMI for consistency
data["BMI"] = data["Weight_kg"] / ((data["Height_cm"] / 100) ** 2)

# Encode categorical columns
le_gender = LabelEncoder()
le_disease = LabelEncoder()
le_goal = LabelEncoder()

data["Gender"] = le_gender.fit_transform(data["Gender"])
data["Disease_Type"] = le_disease.fit_transform(data["Disease_Type"])
data["Goal"] = le_goal.fit_transform(data["Goal"])

# Features and target
X = data[["Age", "Gender", "Weight_kg", "Height_cm", "BMI", "Disease_Type"]]
y = data["Goal"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42)
}

# Train and evaluate models
accuracies = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.2f}")

# Select best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"\n✅ Best Model: {best_model_name} with accuracy {accuracies[best_model_name]:.2f}")

# Save best model and encoders
joblib.dump({
    "model": best_model,
    "scaler": scaler,
    "le_gender": le_gender,
    "le_disease": le_disease,
    "le_goal": le_goal
}, "diet_model.pkl")

print("✅ Model saved successfully as diet_model.pkl")
