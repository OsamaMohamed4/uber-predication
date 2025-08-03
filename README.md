
# 🚕 NYC Taxi Fare Prediction

## 📌 Objective

The goal of this project is to analyze a dataset of New York City taxi rides and build machine learning models to **predict the fare amount**. This dataset gives useful insights for understanding the factors affecting taxi fares — such as weather, distance, driver, and car condition.

---

## 📂 Dataset Overview

The dataset contains the following columns:

- `User ID`: Unique ID for each user  
- `User Name`: Name of the customer  
- `Driver Name`: Name of the driver  
- `Car Condition`: Current state of the taxi (e.g., good, average)  
- `Weather`: Weather condition during the ride  
- `Pickup Location`: Where the ride started  
- `Drop Location`: Where the ride ended  
- `Distance (in km)`: Distance covered in the trip  
- `Fare`: Target value (in USD 💵)

---

## 🧠 What I Did

1. **Data Cleaning**  
   - Removed missing values  
   - Made sure all data types were correct

2. **Data Encoding**  
   - Categorical columns were encoded to be used in ML models  
   - Applied Label Encoding where needed

3. **Model Training**  
   - Used multiple regression models:
     - Linear Regression  
     - Decision Tree  
     - Random Forest  
     - Gradient Boosting Regressor

4. **Evaluation**  
   - Compared models using R² Score and Mean Squared Error  
   - Visualized model predictions

---

## Results

The **Random Forest** and **Gradient Boosting** models gave the best results in terms of accuracy. The models could predict fares with good precision based on the input features.

---

## 🛠️ Tech Stack

- Python 
- Pandas / NumPy  
- Scikit-learn  
- Matplotlib / Seaborn  
- Google Colab

---

##  How to Run

1. Download or clone the repo  
2. Open `final_internship.ipynb` using Jupyter Notebook or Google Colab  
3. Run cells step by step to train and evaluate the models
