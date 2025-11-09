import streamlit as st
import pandas as pd
import plotly.express as px
from genai.insight_generator import generate_insights

st.title("Sales Insights & Demand Forecasting Assistant")

# Load data
data = pd.read_csv("data/sales_data.csv")
st.dataframe(data.head())

# Revenue trend visualization
fig = px.line(
    data.groupby("date")["revenue"].sum().reset_index(),
    x="date", 
    y="revenue", 
    title="Revenue Over Time"
)
st.plotly_chart(fig)

# AI-generated insights
forecast_summary = "Next 30 days show a 12% increase in total sales."
metrics = {
    "total_revenue": data["revenue"].sum(),
    "avg_order_value": data["revenue"].mean()
}

ai_insight = generate_insights(metrics, forecast_summary)

st.subheader("AI-Generated Business Insight")
st.write(ai_insight)
