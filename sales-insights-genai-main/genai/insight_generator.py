import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_insights(metrics, forecast_summary):
    prompt = f"""
    You are a data analyst. Based on the metrics below, generate a 5-sentence 
    business insight summary:
    
    Metrics: {metrics}
    Forecast Summary: {forecast_summary}
    
    Provide actionable recommendations to improve future sales.
    """
    
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    
    return response.text
