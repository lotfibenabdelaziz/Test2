import requests
import pandas as pd

# Flask API URL
url = "http://127.0.0.1:5000/predict"

# Define multiple test cases
test_inputs = [
    {
        "Hour": 9, "Temperature": 18.5, "Humidity": 60, "Wind_speed": 1.5, "Visibility": 1200,
        "Dew_point_temperature": 8.0, "Solar_Radiation": 1.2, "Rainfall": 0.0, "Snowfall": 0.0,
        "Seasons": "Summer", "Holiday": "No Holiday", "Functioning_Day": "Yes",
        "is_Holiday_WorkingDay": "No", "is_clear_weather": "Yes", "is_rainy_weather": "No",
        "is_snowy_weather": "No", "Month": 7, "Day": 14, "Weekday": 1, "DayOfYear": 196
    },
    {
        "Hour": 17, "Temperature": 25.0, "Humidity": 45, "Wind_speed": 3.0, "Visibility": 1500,
        "Dew_point_temperature": 12.0, "Solar_Radiation": 2.0, "Rainfall": 0.0, "Snowfall": 0.0,
        "Seasons": "Summer", "Holiday": "No Holiday", "Functioning_Day": "Yes",
        "is_Holiday_WorkingDay": "No", "is_clear_weather": "Yes", "is_rainy_weather": "No",
        "is_snowy_weather": "No", "Month": 7, "Day": 14, "Weekday": 1, "DayOfYear": 196
    },
    {
        "Hour": 7, "Temperature": 5.0, "Humidity": 80, "Wind_speed": 2.0, "Visibility": 800,
        "Dew_point_temperature": 2.0, "Solar_Radiation": 0.5, "Rainfall": 1.0, "Snowfall": 0.0,
        "Seasons": "Winter", "Holiday": "Holiday", "Functioning_Day": "No",
        "is_Holiday_WorkingDay": "Yes", "is_clear_weather": "No", "is_rainy_weather": "Yes",
        "is_snowy_weather": "No", "Month": 12, "Day": 25, "Weekday": 2, "DayOfYear": 359
    }
]

# Collect results
results = []

for idx, input_data in enumerate(test_inputs, start=1):
    response = requests.post(url, json=input_data)
    if response.status_code == 200:
        pred = response.json().get("prediction")
        results.append({**input_data, "Predicted_Bike_Count": pred})
    else:
        print(f"Error in test case {idx}: {response.json()}")

# Convert results to DataFrame for easy display
df_results = pd.DataFrame(results)

# Display table
print("\n📊 Predictions for test cases:")
print(df_results)

# Optionally, save to CSV
df_results.to_csv("predictions_test_cases.csv", index=False)
print("\n✅ Predictions saved to predictions_test_cases.csv")
