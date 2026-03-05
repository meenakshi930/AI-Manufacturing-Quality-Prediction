# AI-Driven Manufacturing Quality Prediction & Defect Prevention Platform

## Project Overview

This project focuses on predicting potential machine failures in a manufacturing environment using Machine Learning. By analyzing important machine parameters such as temperature, rotational speed, torque, and tool wear, the system predicts whether a machine is likely to fail.

The goal of this project is to help manufacturing industries detect possible defects early and prevent production issues, improving product quality and reducing downtime.

---

## Dataset

Dataset used: **AI4I 2020 Predictive Maintenance Dataset**

The dataset contains sensor and operational data from industrial machines.

### Features Used

* Air temperature [K]
* Process temperature [K]
* Rotational speed [rpm]
* Torque [Nm]
* Tool wear [min]

### Target Variable

* **Machine Failure**

  * 0 → No Failure
  * 1 → Failure

---

## Machine Learning Model

The machine learning algorithm used in this project is:

**Random Forest Classifier**

Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their results to improve prediction accuracy and reduce overfitting.

### Model Performance

* **Accuracy:** ~98%

---

## Backend Implementation

The backend is implemented using **Python and Flask**.

The backend performs the following tasks:

* Loads the trained machine learning model
* Accepts machine parameters from the frontend
* Processes the input data
* Returns the predicted result (Failure / No Failure)

### API Endpoint

```
POST /predict
```

### Example Request

```json
{
  "air_temperature": 298,
  "process_temperature": 308,
  "rotational_speed": 1500,
  "torque": 40,
  "tool_wear": 10
}
```

### Example Response

```json
{
  "prediction": "No Failure"
}
```

---

## System Workflow

1. User enters machine parameters in the frontend interface.
2. The frontend sends a POST request to the backend API.
3. The backend processes the input data.
4. The trained Random Forest model predicts machine failure.
5. The prediction result is returned to the frontend and displayed to the user.

---

## Project Structure

```
AI-Manufacturing-Quality-Prediction
│
├── backend
│   ├── app.py
│   ├── data
│   │   └── raw
│   │       └── ai4i2020.csv
│   │
│   ├── models
│   │   ├── model.pkl
│   │   └── scaler.pkl
│   │
│   └── src
│       ├── preprocess.py
│       └── train_model.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation & Setup

### 1. Clone the Repository

```
git clone https://github.com/meenakshi930/AI-Manufacturing-Quality-Prediction.git
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run the Backend Server

```
python backend/app.py
```

The server will start at:

```
http://127.0.0.1:5000
```

---

## Technologies Used

* Python
* Flask
* Scikit-Learn
* Pandas
* NumPy
* Git & GitHub

---

## Contributors

**Backend Development:**
Meenakshi Gupta

**Frontend Development:**
Anshika Garg

---

## Future Improvements

* Improve model performance using advanced algorithms
* Deploy the system on cloud platforms
* Integrate real-time sensor data
* Add visualization dashboard for monitoring machine health