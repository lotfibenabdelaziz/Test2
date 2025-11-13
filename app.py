from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("rf_model.pkl")

@app.route("/")
def home():
    return jsonify({
        "message": "🚴‍♂️ Bike Rental Prediction API",
        "usage": "Send a POST request to /predict with feature values in JSON format."
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract data from JSON
        hour = data.get("Hour", 12)
        temp = data.get("Temperature", 20.0)
        humidity = data.get("Humidity", 50.0)
        wind = data.get("Wind_speed", 2.0)
        visibility = data.get("Visibility", 1000.0)
        dew_point = data.get("Dew_point_temperature", 10.0)
        solar = data.get("Solar_Radiation", 1.5)
        rainfall = data.get("Rainfall", 0.0)
        snowfall = data.get("Snowfall", 0.0)
        season = data.get("Seasons", "Spring")
        holiday = data.get("Holiday", "No Holiday")
        functioning_day = data.get("Functioning_Day", "Yes")
        is_holiday_workingday = data.get("is_Holiday_WorkingDay", "No")
        is_clear_weather = data.get("is_clear_weather", "No")
        is_rainy_weather = data.get("is_rainy_weather", "No")
        is_snowy_weather = data.get("is_snowy_weather", "No")
        month = data.get("Month", 6)
        day = data.get("Day", 15)
        weekday = data.get("Weekday", 2)
        dayofyear = data.get("DayOfYear", 166)

        # Encoding maps
        season_map = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
        holiday_map = {"No Holiday": 0, "Holiday": 1}
        func_map = {"Yes": 1, "No": 0}
        binary_map = {"No": 0, "Yes": 1}

        # Encoded values
        season_val = season_map.get(season, 0)
        holiday_val = holiday_map.get(holiday, 0)
        function_val = func_map.get(functioning_day, 1)
        holiday_working_val = binary_map.get(is_holiday_workingday, 0)
        clear_weather_val = binary_map.get(is_clear_weather, 0)
        rainy_weather_val = binary_map.get(is_rainy_weather, 0)
        snowy_weather_val = binary_map.get(is_snowy_weather, 0)

        # Build input DataFrame
        input_df = pd.DataFrame([[ 
            hour, temp, humidity, wind, visibility, dew_point, solar, rainfall, snowfall,
            season_val, holiday_val, function_val,
            holiday_working_val, clear_weather_val, rainy_weather_val, snowy_weather_val,
            month, day, weekday, dayofyear
        ]], columns=[
            'Hour', 'Temperature', 'Humidity', 'Wind_speed', 'Visibility', 'Dew_point_temperature',
            'Solar_Radiation', 'Rainfall', 'Snowfall',
            'Seasons', 'Holiday', 'Functioning_Day',
            'is_Holiday_WorkingDay', 'is_clear_weather', 'is_rainy_weather', 'is_snowy_weather',
            'Month', 'Day', 'Weekday', 'DayOfYear'
        ])

        # Prediction
        prediction = model.predict(input_df)[0]

        return jsonify({
            "prediction": int(prediction),
            "input_data": data
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
