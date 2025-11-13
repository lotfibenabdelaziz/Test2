
# 🚴‍♂️ Bike Rental Demand Prediction

## **Project Overview**

Predict the number of bikes rented based on **time, weather, and daily features**. This project helps bike-sharing operators optimize fleet allocation, prepare for peak demand, and improve operational efficiency.

The project uses **Python**, **scikit-learn**, **Flask**, **Docker**, and **Kubernetes** for deployment.

---

## **Dataset**

**Source:** [Kaggle]

**Columns / Features:**

| Feature               | Type  | Description                            |
| --------------------- | ----- | -------------------------------------- |
| Hour                  | int   | Hour of the day (0–23)                 |
| Temperature           | float | °C                                     |
| Humidity              | float | %                                      |
| Wind_speed            | float | m/s                                    |
| Visibility            | float | 10m units                              |
| Dew_point_temperature | float | °C                                     |
| Solar_Radiation       | float | MJ/m²                                  |
| Rainfall              | float | mm                                     |
| Snowfall              | float | cm                                     |
| Seasons               | int   | 0=Spring, 1=Summer, 2=Autumn, 3=Winter |
| Holiday               | int   | 0=No Holiday, 1=Holiday                |
| Functioning_Day       | int   | 0=No, 1=Yes                            |
| is_Holiday_WorkingDay | int   | Binary flag                            |
| is_clear_weather      | int   | Binary flag                            |
| is_rainy_weather      | int   | Binary flag                            |
| is_snowy_weather      | int   | Binary flag                            |
| Month                 | int   | Month extracted from date              |
| Day                   | int   | Day of month                           |
| Weekday               | int   | Day of week (0=Monday)                 |
| DayOfYear             | int   | Day of year                            |

**Target:** `Rented_Bike_Count`

---

## **Environment Setup**

### **1️⃣ Clone the repository**

```bash
git https://github.com/lotfibenabdelaziz/Test2.git
cd Test2
```

### **2️⃣ Install Python dependencies**

```bash
pip install -r requirements.txt
```

---

## **Running Locally**

### **Flask API**

```bash
python app.py
```

* API endpoint: `http://127.0.0.1:5000/predict`
* Send a POST request with **JSON input data** to get predictions.

**Example request using Python `requests`:**

```python
import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "Hour": 10,
    "Temperature": 22,
    "Humidity": 55,
    "Wind_speed": 2.0,
    "Visibility": 1200,
    "Dew_point_temperature": 9,
    "Solar_Radiation": 1.2,
    "Rainfall": 0.0,
    "Snowfall": 0.0,
    "Seasons": 1,
    "Holiday": 0,
    "Functioning_Day": 1,
    "is_Holiday_WorkingDay": 0,
    "is_clear_weather": 1,
    "is_rainy_weather": 0,
    "is_snowy_weather": 0,
    "Month": 7,
    "Day": 14,
    "Weekday": 1,
    "DayOfYear": 196
}

response = requests.post(url, json={"input_data": data})
print(response.json())
```

---

### **Docker**

```bash
docker build -t bike-rental-api .
docker run -p 5000:5000 bike-rental-api
```

* Flask API will be accessible at `http://localhost:5000/predict`.

---

## **Recommended 3-Terminal Workflow**

| Terminal | Command                   | Purpose                                        |
| -------- | ------------------------- | ---------------------------------------------- |
| 1        | `python app.py`           | Run Flask API for predictions                  |
| 2        | `pytest test_training.py` | Automated tests for model training pipeline    |
|3          |`pytest test_api.py ` |Test API endpoints with sample JSON requests  |

This ensures your **API, tracking, and tests** are synchronized.

---

## **Training Pipeline**

1. Preprocess data (feature extraction, encoding).
2. Split into `X` (features) and `y` (target).
3. Train **Random Forest Regressor**.
4. Log metrics and artifacts with **MLflow**.
5. Save trained model with **joblib**.
6. Generate **feature importance plots**.

**Tools used:**

* `pandas`, `numpy` → data processing
* `scikit-learn` → Random Forest model
* `matplotlib` → plotting
* `joblib` → save/load models
* `Flask` → REST API

---

## **CI / GitHub Actions**

* Automatically runs tests on every push to `main`.
* `.github/workflows/ci.yml` can:

  * Install dependencies
  * Run `pytest`
  * Build Docker image
  * Simulate Kubernetes deployment using `kubectl --dry-run=client -f k8s/`

**Local CI Testing:**

```bash
cd Test
pytest test_training.py -v
```

---

## **Git Commands**

```bash
git status       
git add .        
git commit -m "Add feature / fix bug"
git push origin main
git pull origin main
```

---

## **References**

* [Pandas Documentation](https://pandas.pydata.org/)
* [NumPy Documentation](https://numpy.org/)
* [Scikit-learn Documentation](https://scikit-learn.org/)
* [Matplotlib Documentation](https://matplotlib.org/)
* [Flask Documentation](https://flask.palletsprojects.com/)
* [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
* [Docker Documentation](https://docs.docker.com/)
* [Kubernetes Documentation](https://kubernetes.io/docs/home/)

