import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# minimal dummy dataset to test training pipeline
@pytest.fixture
def sample_data():
    # 10 rows of dummy data with the same columns as your X
    data = pd.DataFrame({
        'Hour': np.random.randint(0, 24, 10),
        'Temperature': np.random.uniform(0, 40, 10),
        'Humidity': np.random.uniform(0, 100, 10),
        'Wind_speed': np.random.uniform(0, 10, 10),
        'Visibility': np.random.uniform(0, 2000, 10),
        'Dew_point_temperature': np.random.uniform(-10, 30, 10),
        'Solar_Radiation': np.random.uniform(0, 5, 10),
        'Rainfall': np.random.uniform(0, 50, 10),
        'Snowfall': np.random.uniform(0, 10, 10),
        'Seasons': np.random.randint(0, 4, 10),
        'Holiday': np.random.randint(0, 2, 10),
        'Functioning_Day': np.random.randint(0, 2, 10),
        'is_Holiday_WorkingDay': np.random.randint(0, 2, 10),
        'is_clear_weather': np.random.randint(0, 2, 10),
        'is_rainy_weather': np.random.randint(0, 2, 10),
        'is_snowy_weather': np.random.randint(0, 2, 10),
        'Month': np.random.randint(1, 13, 10),
        'Day': np.random.randint(1, 29, 10),
        'Weekday': np.random.randint(0, 7, 10),
        'DayOfYear': np.random.randint(1, 366, 10),
        'Rented_Bike_Count': np.random.randint(0, 500, 10)
    })
    return data

def test_training_pipeline(sample_data):
    # Split features and target
    X = sample_data.drop(columns=['Rented_Bike_Count'])
    y = sample_data['Rented_Bike_Count']
    
    # Train model
    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Predict
    y_pred = model.predict(X)
    
    # Assertions
    assert y_pred.shape == y.shape, "Predictions shape mismatch"
    assert np.all(y_pred >= 0), "Predictions contain negative values"
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    # Metrics should be finite
    assert np.isfinite(mae), "MAE is not finite"
    assert np.isfinite(rmse), "RMSE is not finite"
    assert np.isfinite(r2), "R2 is not finite"
