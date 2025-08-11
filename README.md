
 🚗 Bengaluru Smart Parking Availability Predictor 🛵

This Streamlit web application predicts the availability of **4-wheeler** and **2-wheeler** parking slots across major Bengaluru locations using machine learning models trained on historical data.


## 📊 Features

- Predicts parking availability based on:
  - Day of the week
  - Time slot (start & end)
  - Location
  - Weather conditions
  - Whether it is a holiday
  - Nearby events
- Uses two machine learning models:
  - **Random Forest**
  - **XGBoost**
- Automatically fills total parking slots based on selected location
- Simple and interactive UI using Streamlit


## 🛠️ Tech Stack

- Python
- Streamlit
- scikit-learn
- XGBoost
- NumPy, Pandas
- Joblib (for model serialization)


## 🧠 Models

Both models were trained to predict the number of available:
- 4-Wheeler slots
- 2-Wheeler slots

The models were trained on features like:
- `day_of_week`, `location`, `is_holiday`, `weather_condition`, `total_slots`, `event_nearby`, `start_time`, `end_time`

All categorical features are label-encoded, and numeric features are scaled using `StandardScaler`.


## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/bangalore-parking-predictor.git
   cd bangalore-parking-predictor


2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Make sure these files are present in the directory**:

   * `rf_model.pkl`
   * `xgb_model.pkl`
   * `scaler.pkl`
   * `encoder.pkl`
   * `App.py`

4. **Run the Streamlit app**

   ```bash
   streamlit run App.py
   ```


## 📸 Sample UI

> A friendly interface for predicting smart parking availability. Choose time, location, and other options from the sidebar and see real-time predictions!


## ✍️ Author

* **Prince Kumar Gupta**
  *Aspiring Machine Learning Engineer & Data Analyst*


## 📌 Note

* Negative predictions are clipped to zero using `max(0, prediction)`.
* If the encoder is missing for any categorical feature, an error message is shown.
* You must use the same feature order and transformations as during model training.


## 📬 Feedback

If you find this useful or have ideas for improvement, feel free to create an issue or reach out! 💬

