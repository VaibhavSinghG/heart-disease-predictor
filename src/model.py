import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import pickle

# Load dataset
df = pd.read_csv(r"C:\Users\pc\OneDrive\Documents\java prog\XAI_Heart_Disease_Prediction\data\heart.csv")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#
# Model train hone ke baad:
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Correct way to save model
with open("C:/Users/pc/OneDrive/Documents/java prog/XAI_Heart_Disease_Prediction/model/heart_disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

