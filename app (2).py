# app.py (final fixed version)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random

# Load model and encoders
model_data = joblib.load("diet_model.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
le_gender = model_data["le_gender"]
le_disease = model_data["le_disease"]
le_goal = model_data["le_goal"]

# Load food dataset
food_data = pd.read_csv("food_dataset.csv")
food_data["Allergens"] = food_data["Allergens"].fillna("None").replace(
    ["nan", "NaN", "NONE", " none", "None "], "None"
)

st.set_page_config(page_title="AI Diet Recommendation", layout="wide")

# Sidebar inputs
st.sidebar.header("Enter Your Details")

age = st.sidebar.slider("Age", 10, 80, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
weight = st.sidebar.slider("Weight (kg)", 30, 150, 60)
height = st.sidebar.slider("Height (cm)", 120, 210, 170)
disease = st.sidebar.selectbox("Disease Type", ["None", "Diabetes", "Cholesterol"])
food_pref = st.sidebar.selectbox("Food Preference", ["Both", "Veg", "Non-Veg"])
allergy = st.sidebar.selectbox("Allergy", ["None", "Peanuts", "Gluten", "Lactose"])

# Calculate BMI
bmi = round(weight / ((height / 100) ** 2), 2)

# Safe encoding
try:
    gender_encoded = le_gender.transform([gender])[0]
except Exception:
    gender_encoded = 0

try:
    disease_encoded = le_disease.transform([disease])[0]
except ValueError:
    if "None" in list(le_disease.classes_):
        disease_encoded = int(le_disease.transform(["None"])[0])
    else:
        disease_encoded = 0

# Prepare input
input_df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender_encoded],
    "Weight_kg": [weight],
    "Height_cm": [height],
    "BMI": [bmi],
    "Disease_Type": [disease_encoded]
})

input_scaled = scaler.transform(input_df)
goal_encoded = model.predict(input_scaled)[0]
goal = le_goal.inverse_transform([goal_encoded])[0]

# --- Output Section ---
st.title("💪 Personalized Diet Recommendation System")
st.markdown("### Based on your health details:")

col1, col2 = st.columns(2)
with col1:
    st.metric("Your BMI", f"{bmi}")
with col2:
    st.metric("Predicted Goal", goal)

# --- Food Recommendation Logic ---
filtered_food = food_data.copy()

filtered_food["Food_Type"] = filtered_food["Food_Type"].astype(str).str.strip()
filtered_food["Allergens"] = filtered_food["Allergens"].astype(str).str.strip()
filtered_food["Suitable_For"] = filtered_food["Suitable_For"].astype(str).str.strip()

# Disease filter
if disease != "None":
    filtered_food = filtered_food[filtered_food["Suitable_For"].str.contains(disease, case=False, na=False)]

# Goal filter
filtered_food = filtered_food[filtered_food["Suitable_For"].str.contains(goal, case=False, na=False)]

# Allergy filter
if allergy != "None":
        filtered_food = filtered_food[~filtered_food["Allergens"].str.contains(allergy, case=False, na=False)]

# Food preference handling
if food_pref == "Veg":
    filtered_food = filtered_food[filtered_food["Food_Type"].str.lower() == "veg"]
elif food_pref == "Non-Veg":
    filtered_food = filtered_food[filtered_food["Food_Type"].str.lower() == "non-veg"]
elif food_pref == "Both":
    veg_items = filtered_food[filtered_food["Food_Type"].str.lower() == "veg"]
    nonveg_items = filtered_food[filtered_food["Food_Type"].str.lower() == "non-veg"]
    half_count = 4  # Half veg, half non-veg (total 8 items)
    veg_sample = veg_items.sample(n=min(half_count, len(veg_items)), random_state=42)
    nonveg_sample = nonveg_items.sample(n=min(half_count, len(nonveg_items)), random_state=42)
    filtered_food = pd.concat([veg_sample, nonveg_sample])

# Ensure at least 8 items
def fill_to_minimum(current_df, master_df, min_items=8):
    if len(current_df) >= min_items:
        return current_df.head(min_items)
    needed = min_items - len(current_df)

    pool = master_df.copy()
    if food_pref == "Veg":
        pool = pool[pool["Food_Type"].str.lower() == "veg"]
    elif food_pref == "Non-Veg":
        pool = pool[pool["Food_Type"].str.lower() == "non-veg"]

    if allergy != "None":
        pool = pool[~pool["Allergens"].str.contains(allergy, case=False, na=False)]

    pool = pool[~pool["Food_Item"].isin(current_df["Food_Item"])]

    if len(pool) < needed:
        pool = master_df.copy()
        if allergy != "None":
            pool = pool[~pool["Allergens"].str.contains(allergy, case=False, na=False)]
        pool = pool[~pool["Food_Item"].isin(current_df["Food_Item"])]

    if len(pool) == 0:
        return current_df.head(min_items)

    extra = pool.sample(n=min(needed, len(pool)), replace=False, random_state=42)
    result = pd.concat([current_df, extra]).drop_duplicates(subset="Food_Item").head(min_items)
    return result

filtered_food = fill_to_minimum(filtered_food, food_data, min_items=8)
filtered_food = filtered_food.reset_index(drop=True)

# Display food table (added Suitable_For & Allergens)
st.markdown("### 🍽️ Recommended Food Items")
st.dataframe(
    filtered_food[[
        "Food_Item", "Calories", "Protein_g", "Fat_g",
        "Carbohydrate_g", "Food_Type", "Suitable_For", "Allergens"
    ]],
    use_container_width=True
)

st.markdown("---")
