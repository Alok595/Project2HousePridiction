


# 🏡 Housing Price Prediction

## 📌 Project Overview

This project implements a **Machine Learning model** using **Linear Regression** to predict housing prices.
The prediction is based on multiple features such as:

* 🏠 **Area (sqft)**
* 🛏️ **Bedrooms**
* 📍 **Location**
* 🚿 **Bathrooms**
* 🪑 **Furnishing Status**
* 🚗 **Parking Availability**
* 🏗️ **Age of Property**
* 🌿 **Balcony Count**

The model is trained to estimate **house prices (in Lakhs)** and provides valuable insights through **visualizations**.

---

## 🎯 Objective

To build a predictive model that can accurately estimate house prices from given property details, enabling better decision-making for buyers, sellers, and real estate professionals.

---

## 📊 Dataset

* Contains **1000 rows** of housing data.
* Features include both **numerical** and **categorical** attributes.
* Preprocessing steps:

  * Handling missing values
  * Encoding categorical variables (One-Hot Encoding for `Location`)
  * Converting datatypes where necessary

---

## ⚙️ Tech Stack & Environment

* **Language:** Python 3.x
* **Libraries:**

  * `pandas` → Data manipulation
  * `numpy` → Numerical operations
  * `matplotlib` & `seaborn` → Visualization
  * `scikit-learn` → Machine Learning (Linear Regression, Train-Test Split, Evaluation Metrics)

---

## 🚀 How to Run

1. Clone this repository or download the notebook/script.

2. Install required dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. Place the dataset (`sample_housing_dataset_1000_rows.csv`) in the project directory.

4. Open the notebook:

   ```bash
   jupyter notebook Housing_Price_Prediction.ipynb
   ```

5. Run all cells sequentially to:

   * Preprocess data
   * Train the model
   * Evaluate performance
   * Generate visualizations
   * Predict housing prices for new inputs

---

## 📈 Results & Insights

* **Model Used:** Linear Regression

* **Evaluation Metrics:**

  * R² Score → Measures model accuracy
  * Mean Squared Error (MSE) → Measures prediction error

* **Visualizations Generated:**

  * Distribution of **Area** & **Price**
  * Price comparison by **Location** & **Bedrooms**
  * Boxplot showing **Price distribution across top locations**

➡️ The model successfully predicts house prices and demonstrates strong potential for real-world applications.

---

## 📂 Dataset Source

A **sample housing dataset** with \~1000 rows containing property details and prices.

---

## 🔮 Future Enhancements

* Experiment with advanced models (Random Forest, XGBoost, Gradient Boosting).
* Perform hyperparameter tuning for better accuracy.
* Deploy the model using **Flask / FastAPI** or as a **web app (Streamlit/Dash)**.
* Enrich dataset with additional features (nearby amenities, property type, floor number, etc.).

---

✨ **Author:** *Alok Ranjan*
📅 **Submission Date:** *21 August 2025*

---

