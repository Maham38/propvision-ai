<img width="975" height="269" alt="image" src="https://github.com/user-attachments/assets/9f9f422d-1314-4ec2-867d-c518cf9b4378" /># 🏠 PropVision AI  
## An End-to-End Machine Learning Framework for House Price Prediction

PropVision AI is a machine learning-based house price prediction system developed using real-world property listings from **Zameen.com** for Islamabad. 

The project focuses on the complete machine learning pipeline, including web scraping, data collection, preprocessing, feature engineering, regression model development, performance comparison, and deployment of a prediction system.

The system predicts estimated house prices based on important property features such as area, location, bedrooms, bathrooms, property type, and additional facilities.

---

# 📌 Project Objective

The objective of this project is to develop an **End-to-End House Price Prediction System** that can learn from real estate listing data and estimate property prices using machine learning regression techniques.

The project covers:

- Scraping property listings from Zameen.com
- Storing collected data in CSV format
- Cleaning and preprocessing property data
- Performing feature engineering
- Training multiple regression models
- Comparing model performance
- Developing a user-friendly prediction system

---

# 🚀 Project Workflow

```
Zameen.com Property Listings

            ↓

Web Scraping

            ↓

CSV Dataset Creation

            ↓

Data Cleaning & Preprocessing

            ↓

Feature Engineering

            ↓

Machine Learning Model Training

            ↓

Model Evaluation & Comparison

            ↓

Best Model Selection

            ↓

House Price Prediction System
```

---

# 📊 Dataset Description

The dataset consists of **350 Islamabad property listings** collected from Zameen.com.

Each record represents a residential property with multiple features used for prediction.

## Dataset Features

| Feature | Description |
|---------|-------------|
| Price | Target variable (House price in PKR) |
| Area | Property size |
| City | Property city |
| Bedrooms | Number of bedrooms |
| Bathrooms | Number of bathrooms |
| Location | Property locality |
| Property Type | House, Flat, Portion, Farm House, etc. |
| Built in Year | Construction year |
| Parking Space | Number of parking spaces |
| Servant Quarters | Availability/count of servant rooms |
| Store Rooms | Number of store rooms |
| Kitchens | Number of kitchens |
| Drawing Rooms | Number of drawing rooms |

---

# 🕸️ Data Collection & Web Scraping

## Data Acquisition

Property listing data was collected from **Zameen.com** for Islamabad properties.

The scraping process extracts:

- Property price
- Area
- Location
- Property type
- Rooms information
- Additional property features

The collected data is stored in CSV format for further processing.

### Technologies Used

- Python
- BeautifulSoup
- Requests
- Pandas

---

# 🧹 Data Preprocessing & Feature Engineering

Before training machine learning models, the dataset was prepared using the following steps:

### Data Cleaning

- Handling missing values
- Removing duplicate records
- Removing inconsistent data entries

### Feature Transformation

- Converting price values into numeric format
- Standardizing area measurements
- Encoding categorical variables:
  - Location
  - Property type

### Dataset Preparation

- Splitting data into:
  - Training dataset
  - Testing dataset

- Building preprocessing pipelines using Scikit-learn

---

# 🤖 Machine Learning Model Development

Six regression algorithms were implemented and evaluated:

## 1. Linear Regression

A baseline regression algorithm used to understand the relationship between property features and prices.

## 2. Decision Tree Regressor

A tree-based model that learns decision rules from property features.

## 3. Random Forest Regressor

An ensemble learning method that combines multiple decision trees to improve prediction accuracy.

## 4. Gradient Boosting Regressor

A boosting algorithm that builds models sequentially to reduce prediction errors.

## 5. XGBoost Regressor

An optimized gradient boosting algorithm designed for structured datasets.

## 6. CatBoost Regressor

A powerful boosting algorithm that efficiently handles categorical features.

---

# 📈 Model Evaluation

All models were evaluated using standard regression metrics:

### Mean Absolute Error (MAE)

Measures the average prediction error.

### Mean Squared Error (MSE)

Measures squared prediction differences.

### Root Mean Squared Error (RMSE)

Measures prediction error magnitude.

### R² Score

Shows how well the model explains variations in house prices.

---

# 🏆 Best Performing Model

After comparing all regression algorithms:

## Gradient Boosting Regressor

was selected as the best-performing model.

Performance:

- Highest R² Score
- Lowest prediction errors
- Better generalization compared to other models

Final Model Accuracy:

```
R² Score: 0.9436
```
<img width="975" height="269" alt="image" src="https://github.com/user-attachments/assets/b6348d79-6fe1-4c1a-938a-f724a6ab825f" />

---

# 🖥️ Final Prediction System

A Streamlit-based web application was developed as the final system.

Users can enter:

- Area
- Bedrooms
- Bathrooms
- Location
- Property Type
- Parking Spaces
- Kitchens
- Store Rooms
- Drawing Rooms
- Servant Quarters

The system outputs:

```
Estimated House Price (PKR)
```

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Data Collection

- BeautifulSoup
- Requests

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- XGBoost
- CatBoost

## Model Storage

- Joblib

## Application Development

- Streamlit

---

# 📂 Project Structure

```
PropVision-AI/

│
├── data/
│   └── islamabad_listings.csv
│
├── models/
│   ├── best_house_price_model.joblib
│   └── model_metrics.csv
│
├── src/
│   ├── scraper.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── generate_dataset.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation & Setup

## 1. Clone Repository

```bash
git clone https://github.com/your-username/PropVision-AI.git

cd PropVision-AI
```

---

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

### Windows

```bash
.venv\Scripts\activate
```

### Linux/Mac

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Train Machine Learning Models

```bash
python src/train.py
```

This will:

- Load dataset
- Apply preprocessing
- Train all regression models
- Compare performance
- Save the best model

---

## Run Prediction Application

```bash
streamlit run app.py
```

Application will open:

```
http://localhost:8501
```

---

# ⚠️ Challenges Faced

## Website Structure Changes

Real estate websites frequently change their HTML structure, which affects scraping reliability.

## Data Inconsistency

Property data contains different formats for:

- Area units
- Price values
- Locations

Data normalization was required.

## Mixed Data Types

The dataset contains both numerical and categorical features, requiring encoding and transformation.

---
# Output

<img width="975" height="672" alt="image" src="https://github.com/user-attachments/assets/11b315c1-3ff3-4b8b-b4bf-d6b08d03c2a6" />
<img width="975" height="716" alt="image" src="https://github.com/user-attachments/assets/e42a1482-3fe6-4e99-9790-b53b7f47efad" />


---
# 🔮 Future Improvements

- Increase dataset size
- Add more cities of Pakistan
- Implement automated scraping scheduler
- Deploy model as REST API
- Add interactive price analytics dashboard
- Apply advanced hyperparameter tuning

---

# 🎯 Conclusion

PropVision AI successfully implements an end-to-end machine learning framework for house price prediction.

The project fulfills all required stages:

✅ Data acquisition through web scraping  
✅ Dataset creation  
✅ Data preprocessing  
✅ Feature engineering  
✅ Regression model development  
✅ Model evaluation  
✅ Final prediction system  

The Gradient Boosting model achieved the best performance and was selected for predicting Islamabad property prices.

---

# 👨‍💻 Author

**Maham Maryam**  
BS Software Engineering  
COMSATS University Islamabad
