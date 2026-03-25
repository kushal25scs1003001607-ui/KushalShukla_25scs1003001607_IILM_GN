import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Sample dataset (you can replace with real data later)
data = {
    "age": [20, 25, 30, 35, 40, 28, 32, 45, 50, 23],
    "weight": [55, 65, 70, 80, 85, 68, 75, 90, 95, 60],
    "height": [160, 170, 175, 180, 165, 172, 178, 182, 168, 169],
    "steps": [5000, 7000, 8000, 6000, 4000, 7500, 8200, 3000, 2000, 9000],
    "heart_rate": [80, 85, 78, 90, 95, 82, 76, 100, 105, 75],
    "calories_burned": [200, 300, 350, 280, 220, 320, 360, 180, 150, 400]
}

df = pd.DataFrame(data)

# Features and target
X = df[["age", "weight", "height", "steps", "heart_rate"]]
y = df["calories_burned"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
error = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", error)

# ---- User Input ----
print("\nEnter your details:")

age = int(input("Age: "))
weight = float(input("Weight (kg): "))
height = float(input("Height (cm): "))
steps = int(input("Steps walked: "))
heart_rate = int(input("Heart rate: "))

user_data = np.array([[age, weight, height, steps, heart_rate]])

prediction = model.predict(user_data)

print("\n🔥 Estimated Calories Burned:", round(prediction[0], 2))
