import nltk
from typing import Dict
import streamlit as st
import plotly.express as px
import pandas as pd

from utils import load_sentiment_analyzer

sia = load_sentiment_analyzer()

def advanced_risk_assessment(text: str) -> Dict:
    """Enhanced risk assessment with proper error handling"""
    if not text:
        return {
            'categories': {},
            'total_risks': 0,
            'severity_counts': {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
            'total_score': 0
        }

    risk_categories = {
        "Compliance": {
            "keywords": ["regulation", "legal", "gdpr", "hipaa", "violation"],
            "weight": 1.8,
            "severity": "High"
        },
        "Financial": {
            "keywords": ["penalty", "fine", "liability", "indemnity"],
            "weight": 2.2,
            "severity": "Critical"
        },
        "Operational": {
            "keywords": ["termination", "breach", "default", "force majeure"],
            "weight": 1.5,
            "severity": "Medium"
        }
    }

    try:
        sentiment = sia.polarity_scores(text)
        sentences = nltk.sent_tokenize(text)
        avg_sentence_length = sum(len(nltk.word_tokenize(s)) for s in sentences) / len(sentences) if sentences else 0

        risk_results = {
            "categories": {},
            "total_risks": 0,
            "severity_counts": {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
            "total_score": 0
        }

        for category, config in risk_categories.items():
            count = sum(text.lower().count(keyword) for keyword in config["keywords"])
            weighted_score = min(40, count * config["weight"])

            risk_results["categories"][category] = {
                "score": weighted_score,
                "count": count,
                "severity": config["severity"]
            }
            risk_results["total_risks"] += count
            risk_results["severity_counts"][config["severity"]] += count

        # Calculate total score
        risk_results["total_score"] = round(min(100,
                                                  sum([v["score"] for v in risk_results["categories"].values()]) +
                                                  (1 - sentiment['compound']) * 25 +
                                                  min(30, avg_sentence_length * 0.5)
                                                  ), )

        return risk_results
    except Exception as e:
        st.error(f"Risk assessment failed: {str(e)}")
        return {
            'categories': {},
            'total_risks': 0,
            'severity_counts': {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
            'total_score': 0
        }

def visualize_risks(risk_data):
    """Safe visualization generation with error handling"""
    if not risk_data or not risk_data.get('categories'):
        return None, None

    try:
        # Severity distribution pie chart
        fig1 = px.pie(
            names=list(risk_data["severity_counts"].keys()),
            values=list(risk_data["severity_counts"].values()),
            title="Risk Severity Distribution",
            hole=0.3
        )

        # Category scores bar chart
        categories = list(risk_data["categories"].keys())
        scores = [v.get("score", 0) for v in risk_data["categories"].values()]
        counts = [v.get("count", 0) for v in risk_data["categories"].values()]

        fig2 = px.bar(
            x=categories,
            y=scores,
            text=counts,
            title="Risk Scores by Category",
            labels={"x": "Category", "y": "Risk Score"},
            color=categories
        )

        return fig1, fig2
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        return None, None
