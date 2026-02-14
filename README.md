# 🍔 Food Delivery Churn & Coupon Optimization Engine

## 📌 Business Context

Food delivery platforms face high customer churn and inefficient discount allocation. 
Blanket coupon campaigns reduce profit margins without guaranteeing customer retention.

This project builds a data-driven decision engine to:

- Predict customer churn probability
- Quantify revenue at risk
- Recommend targeted discount strategies

---

## 🎯 Objective

To optimize retention marketing by identifying high-risk users and allocating discounts strategically instead of using blanket campaigns.

---

## 🧠 Methodology

### 1️⃣ RFM Analysis
- **Recency** – Days since last order
- **Frequency** – Total number of orders
- **Monetary** – Total spending

### 2️⃣ Churn Definition
Customer considered churned if no order in the last 30 days.

### 3️⃣ Machine Learning Model
- Logistic Regression (Binary Classification)
- Outputs churn probability for each user

### 4️⃣ Coupon Optimization Logic
Rule-based discount allocation:
- > 70% churn probability → 30% discount
- 40–70% → 15% discount
- < 40% → No discount

### 5️⃣ Revenue Impact Simulation
Expected Revenue at Risk = Churn Probability × Average Spend

---

## 📊 Sample Results

- 120 Users Analyzed
- 51 High-Risk Users Identified
- ₹1.03 Lakh Estimated Revenue at Risk
- Targeted discount allocation reduces unnecessary coupon spend

---

## 🛠 Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly

---

## 🚀 Key Business Impact

- Enables targeted retention strategy
- Protects marketing margins
- Converts raw order data into actionable insights
- Demonstrates ML + business integration

---

## ▶️ How to Run

```
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Dataset

Includes synthetic dataset (300+ rows) for demonstration purposes.
