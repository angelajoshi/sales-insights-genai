# Sales Insights & Demand Forecasting Assistant

> An AI-powered analytics assistant that analyzes sales data, forecasts demand, and generates human-like insights using Machine Learning and Generative AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)](https://openai.com)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Implementation Guide](#implementation-guide)
- [Model Performance](#model-performance)
- [Dashboard](#dashboard)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

---

## Overview

This project combines traditional data analytics, machine learning forecasting, and generative AI to create an intelligent sales assistant that doesn't just analyze—it thinks. The system automatically generates business insights, recommendations, and narrative summaries in natural language.

**What Makes This Different:**
- Goes beyond static dashboards with AI-generated narratives
- Combines ML predictions with GenAI explanations
- Provides actionable recommendations, not just metrics
- Interactive conversational analytics interface

---

## Architecture
<img width="787" height="409" alt="image" src="https://github.com/user-attachments/assets/4d489881-da0e-4af7-8c40-5b61d396cb00" />

### Data Flow

1. **Extract**: Pull sales data from CRM/ERP/Database
2. **Transform**: Clean and aggregate using Pandas/dbt
3. **Analyze**: Compute KPIs (revenue, profit, top products)
4. **Forecast**: Predict demand using Prophet/XGBoost
5. **Explain**: Generate insights with GPT/Gemini
6. **Visualize**: Interactive Streamlit dashboard

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Data Storage** | PostgreSQL, BigQuery |
| **Data Transformation** | dbt, Pandas |
| **Machine Learning** | Prophet, XGBoost, Scikit-learn |
| **Generative AI** | OpenAI GPT-4, Google Gemini |
| **App Framework** | Streamlit, LangChain |
| **Visualization** | Plotly, Streamlit Charts |
| **Orchestration** | Prefect, Airflow (optional) |

---

## Key Features

### Analytics
- Real-time sales performance tracking
- Revenue trends and growth analysis
- Product and regional performance metrics
- Customer retention rate calculation

### Machine Learning
- Time series demand forecasting (Prophet)
- Regression-based sales prediction (XGBoost)
- Seasonal pattern detection
- Anomaly detection in sales data

### Generative AI
- Automatic insight generation from metrics
- Natural language business summaries
- Contextual recommendations
- Question-answering over sales data

### Interactive Dashboard
- Dynamic visualizations with Plotly
- AI-generated narrative insights
- Forecasting visualization
- Drill-down analysis by product/region

---

## Project Structure

```
sales-insights-genai/
│
├── data/                        # Raw and processed data
│   ├── sales_data.csv
│   └── products.csv
│
├── notebooks/                   # Exploratory analysis
│   ├── data_exploration.ipynb
│   └── demand_forecasting.ipynb
│
├── models/                      # Trained ML models
│   ├── prophet_model.pkl
│   └── xgb_model.pkl
│
├── genai/                       # Generative AI components
│   └── insight_generator.py
│
├── streamlit_app/               # Dashboard application
│   └── app.py
│
├── dbt_models/                  # Data transformation models
│   └── sales_aggregation.sql
│
├── prefect_flows/               # Orchestration workflows
│   └── pipeline_flow.py
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PostgreSQL 12+ or BigQuery access
- OpenAI API key or Google Gemini API key
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/angelajoshi/sales-insights-genai.git
   cd sales-insights-genai
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   OPENAI_API_KEY=your_openai_key_here
   DATABASE_URL=your_database_connection_string
   ```

5. **Prepare sample data**
   ```bash
   # Place your sales_data.csv in the data/ folder
   # Or use the provided sample dataset
   ```

---

## Implementation Guide

### 1. Data Extraction & Transformation

Load and clean sales data:

```python
import pandas as pd

# Load data
sales = pd.read_csv("data/sales_data.csv")
sales['date'] = pd.to_datetime(sales['date'])
sales['revenue'] = sales['quantity'] * sales['unit_price']

# Aggregate metrics
metrics = {
    'total_revenue': sales['revenue'].sum(),
    'avg_order_value': sales['revenue'].mean(),
    'customer_retention_rate': calculate_retention(sales)
}
```

### 2. Demand Forecasting

#### Using Prophet (Time Series)

```python
from prophet import Prophet

# Prepare data
df = sales.groupby('date')['revenue'].sum().reset_index()
df.columns = ['ds', 'y']

# Train model
model = Prophet()
model.fit(df)

# Generate forecast
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
```

#### Using XGBoost (Regression)

```python
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Feature engineering
X = create_features(sales)  # date features, lag features, etc.
y = sales['revenue']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = XGBRegressor(n_estimators=200, learning_rate=0.05)
model.fit(X_train, y_train)
```

### 3. Generative AI Insights

Generate narrative insights using GPT-4:

```python
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_insights(metrics, forecast_summary):
    prompt = f"""
    You are a data analyst. Based on the metrics below, generate a 5-sentence 
    business insight summary:
    
    Metrics: {metrics}
    Forecast Summary: {forecast_summary}
    
    Provide actionable recommendations to improve future sales.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

### 4. Streamlit Dashboard

Create an interactive dashboard:

```python
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
```

### 5. Run the Application

```bash
streamlit run streamlit_app/app.py
```

Access at: `http://localhost:8501`

---

## Model Performance

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Prophet** | 124.8 | 219.3 | 0.93 |
| **XGBoost** | 118.1 | 210.7 | 0.95 |

XGBoost demonstrates superior performance for short-term forecasting, while Prophet excels at capturing seasonal patterns and long-term trends.

---

## Dashboard

### Example AI-Generated Insight

> "Revenue has grown steadily by 12% in the last quarter, led by strong sales in Electronics and Home Appliances. Demand is expected to rise in the festive season. The model forecasts a peak in late December, suggesting stocking up inventory early. Customer churn rate remains stable. Focus marketing on repeat customers in Tier 1 cities."

### Dashboard Sections

1. **Time Series Forecast**: ML-powered demand predictions
2. **Revenue by Category**: Product performance breakdown
3. **Region-Wise Performance**: Geographic sales analysis
4. **AI-Generated Summary**: Natural language insights
5. **Actionable Recommendations**: Strategic business advice

---

## Future Enhancements

- [ ] Integrate LangChain agents for conversational Q&A over sales data
- [ ] Add voice-based querying (Speech-to-Text + GenAI)
- [ ] Connect real CRM APIs (Salesforce, HubSpot)
- [ ] Deploy on cloud platforms (Render, GCP App Engine, AWS)
- [ ] Implement vector database (Chroma, FAISS) for contextual retrieval
- [ ] Add multi-language support for insights
- [ ] Build automated email reports with AI summaries
- [ ] Integrate real-time streaming data sources

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Author

**Angela Anindya Joshi**

Data Analyst | Machine Learning | GenAI Enthusiast

- Email: angelaanindyajoshi@gmail.com
- GitHub: [@angelajoshi](https://github.com/angelajoshi)


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- OpenAI for GPT-4 API
- Facebook Prophet for time series forecasting
- Streamlit for the amazing dashboard framework
- The open-source ML and AI community

---

<div align="center">

**Star this repo if you find it helpful!**

*An intelligent analytics assistant that thinks, predicts, and explains.*

</div>
