from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from typing import Dict
import re

class RiskAnalyzer:
    def __init__(self):
        nltk.download(['vader_lexicon', 'punkt'])
        self.sia = SentimentIntensityAnalyzer()
        
        self.risk_framework = {
            "Data Protection": {
                "patterns": [r"\b(data breach|unauthorized access)\b"],
                "severity": "High",
                "weight": 1.8
            },
            "Compliance Risk": {
                "patterns": [r"\b(non-compliance|violation|penalty)\b"],
                "severity": "Medium",
                "weight": 1.5
            },
            "Operational Risk": {
                "patterns": [r"\b(downtime|system failure)\b"],
                "severity": "Medium",
                "weight": 1.2
            }
        }

    def analyze_risks(self, text: str) -> Dict:
        """Comprehensive risk analysis with multiple dimensions"""
        results = {
            "categories": {},
            "sentiment": self.sia.polarity_scores(text),
            "complexity": self._calculate_complexity(text),
            "total_score": 0
        }

        # Pattern-based analysis
        category_scores = {}
        for category, config in self.risk_framework.items():
            count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                      for pattern in config["patterns"])
            score = count * config["weight"]
            category_scores[category] = {
                "count": count,
                "score": score,
                "severity": config["severity"]
            }
            results["total_score"] += score

        # Add NLP-based scores
        results["total_score"] += (1 - results["sentiment"]["compound"]) * 20
        results["total_score"] = min(100, results["total_score"])
        
        results["categories"] = category_scores
        return results

    def _calculate_complexity(self, text: str) -> Dict:
        """Calculate document complexity metrics"""
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        return {
            "avg_sentence_length": len(words)/len(sentences) if sentences else 0,
            "unique_words": len(set(words))/len(words) if words else 0,
            "readability_score": len(text)/(len(sentences) + 1)  # Simplified metric
        }