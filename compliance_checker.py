import requests
from bs4 import BeautifulSoup
from typing import Dict
import re

class ComplianceChecker:
    COMPLIANCE_SOURCES = {
        "GDPR": "https://gdpr-info.eu/",
        "HIPAA": "https://www.hhs.gov/hipaa/for-professionals/index.html"
    }

    def __init__(self):
        self.guidelines = self._cache_guidelines()

    def _cache_guidelines(self) -> Dict[str, str]:
        """Cache compliance guidelines with enhanced scraping"""
        guidelines = {}
        for law, url in self.COMPLIANCE_SOURCES.items():
            try:
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                soup = BeautifulSoup(response.text, 'html.parser')
                
                if law == "GDPR":
                    content = soup.find_all(['article', 'section'], limit=8)
                elif law == "HIPAA":
                    content = soup.find_all(['div.content', 'p'], limit=10)
                
                guidelines[law] = "\n".join([elem.get_text().strip() for elem in content if elem.get_text().strip()])
            except Exception as e:
                guidelines[law] = f"Guideline retrieval error: {str(e)}"
        return guidelines

    def check_compliance(self, text: str) -> Dict[str, Dict]:
        """Enhanced compliance check using both patterns and semantic analysis"""
        compliance_status = {
            "GDPR": self._analyze_gdpr(text),
            "HIPAA": self._analyze_hipaa(text)
        }
        return compliance_status

    def _analyze_gdpr(self, text: str) -> Dict[str, str]:
        """GDPR-specific analysis"""
        patterns = {
            "Data Protection Officer": r"\b(DPO|data protection officer)\b",
            "Right to Access": r"\b(right to access|data access)\b",
            "Data Portability": r"\b(data portability)\b",
            "Breach Notification": r"\b(breach notification|72-hour notification)\b"
        }
        return self._check_patterns(text, patterns)

    def _analyze_hipaa(self, text: str) -> Dict[str, str]:
        """HIPAA-specific analysis"""
        patterns = {
            "PHI Protection": r"\b(PHI|protected health information)\b",
            "Access Controls": r"\b(access controls|unique user identification)\b",
            "Audit Controls": r"\b(audit controls|activity review)\b",
            "Transmission Security": r"\b(transmission security|encryption)\b"
        }
        return self._check_patterns(text, patterns)

    def _check_patterns(self, text: str, patterns: Dict) -> Dict[str, str]:
        """Helper method for pattern matching"""
        results = {}
        for requirement, pattern in patterns.items():
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            results[requirement] = {
                'status': "Compliant" if matches else "Not Found",
                'count': len(matches),
                'evidence': matches[:3]  # Show first 3 examples
            }
        return results