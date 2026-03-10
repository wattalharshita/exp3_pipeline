from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib
import os

data = pd.read_csv("data/housing.csv")

X = data[["area", "bedrooms"]]
y = data["price"]

model = LinearRegression()
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/housing_model.pkl")

print("Model trained and saved successfully")
