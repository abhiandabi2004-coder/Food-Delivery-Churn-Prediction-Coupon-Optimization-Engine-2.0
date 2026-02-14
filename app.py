import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px

st.set_page_config(page_title="Food Delivery Churn Engine", layout="wide")

st.title("🍔 Food Delivery Churn & Coupon Optimization Engine")

st.markdown("""
Advanced churn prediction model using behavioral features 
and Random Forest classification.
""")

uploaded_file = st.file_uploader("Upload Order Data CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    required_columns = ["user_id", "order_id", "order_date", "order_value", "discount_given"]

    if not all(col in df.columns for col in required_columns):
        st.error("CSV must contain required columns.")
        st.stop()

    df["order_date"] = pd.to_datetime(df["order_date"])

    snapshot_date = df["order_date"].max()

    # --------------------------
    # FEATURE ENGINEERING
    # --------------------------

    rfm = df.groupby("user_id").agg({
        "order_date": lambda x: (snapshot_date - x.max()).days,
        "order_id": "count",
        "order_value": "sum",
        "discount_given": "sum"
    }).reset_index()

    rfm.columns = ["user_id", "Recency", "Frequency", "Monetary", "Total_Discount"]

    # New Features
    rfm["Avg_Order_Value"] = rfm["Monetary"] / rfm["Frequency"]
    rfm["Discount_Ratio"] = rfm["Total_Discount"] / rfm["Monetary"]

    # --------------------------
    # IMPROVED CHURN DEFINITION
    # --------------------------
    # More realistic: high recency + low frequency

    rfm["Churn"] = np.where(
        (rfm["Recency"] > 45) & (rfm["Frequency"] <= 2),
        1,
        0
    )

    # --------------------------
    # MODEL TRAINING
    # --------------------------

    features = [
        "Recency",
        "Frequency",
        "Monetary",
        "Avg_Order_Value",
        "Discount_Ratio"
    ]

    X = rfm[features]
    y = rfm["Churn"]

    if len(rfm["Churn"].unique()) > 1:

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_scaled = scaler.transform(X)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        rfm["Churn_Probability"] = model.predict_proba(X_scaled)[:, 1]

        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

    else:
        st.warning("Not enough variation in churn data.")
        rfm["Churn_Probability"] = 0
        accuracy = 0
        cm = [[0, 0], [0, 0]]

    # --------------------------
    # SMARTER DISCOUNT STRATEGY
    # --------------------------

    def coupon_strategy(prob):
        if prob > 0.75:
            return 25
        elif prob > 0.50:
            return 15
        elif prob > 0.30:
            return 10
        else:
            return 0

    rfm["Recommended_Discount_%"] = rfm["Churn_Probability"].apply(coupon_strategy)

    # --------------------------
    # REVENUE RISK ESTIMATION
    # --------------------------

    rfm["Expected_Revenue_At_Risk"] = rfm["Churn_Probability"] * rfm["Monetary"]

    # --------------------------
    # DASHBOARD METRICS
    # --------------------------

    st.subheader("📊 Key Business Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Users", len(rfm))
    col2.metric("High Risk Users (>75%)", sum(rfm["Churn_Probability"] > 0.75))
    col3.metric("Estimated Revenue At Risk",
                f"₹ {rfm['Expected_Revenue_At_Risk'].sum():,.0f}")

    # --------------------------
    # MODEL PERFORMANCE
    # --------------------------

    st.subheader("📈 Model Performance")

    st.write(f"Accuracy: {accuracy:.2f}")

    cm_df = pd.DataFrame(
        cm,
        columns=["Predicted No Churn", "Predicted Churn"],
        index=["Actual No Churn", "Actual Churn"]
    )

    st.dataframe(cm_df)

    # --------------------------
    # VISUALIZATION
    # --------------------------

    st.subheader("📉 Churn Probability Distribution")

    fig = px.histogram(
        rfm,
        x="Churn_Probability",
        nbins=20,
        title="Distribution of Churn Probability"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # FINAL TABLE
    # --------------------------

    st.subheader("📋 User-Level Recommendations")

    st.dataframe(
        rfm.sort_values(by="Churn_Probability", ascending=False),
        use_container_width=True
    )