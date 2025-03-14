# 📄 Advanced AI-Driven Legal Document Summarization and Risk Assessment
An advanced AI-powered solution designed to analyze, summarize, and assess risks in legal documents efficiently. It leverages state-of-the-art NLP techniques and specialized legal analysis to assist professionals in making informed decisions faster.

---

## ✨ Key Features

- 📃 **Document Summarization**: Extracts key points and generates concise summaries for lengthy legal documents.
- 🔍 **Risk Evaluation**: Detects potential legal risks, assigns severity levels, and visualizes risk distribution.
- 💬 **Interactive Q&A**: Ask questions related to the document and get precise, context-aware answers.
- 🔀 **Document Comparison**: Highlights differences between two legal documents with structured comparison.
- 📋 **Compliance Analysis**: Identifies essential compliance requirements specific to the document type.
- 📊 **Detailed Reports**: Generates visual PDF reports with graphical representations for better insights.
- 📧 **Email Notifications**: Share document analysis reports via email seamlessly.
- 📜 **Legal Updates**: Keeps track of legal changes that might impact documents.

---

## 🚀 Tech Stack

- **Frontend**: Streamlit
- **AI Models**: LangChain + Groq (Llama 3)
- **Document Processing**: PyMuPDF, NLTK, Regex
- **Vector Search**: FAISS, Sentence Transformers
- **Data Visualization**: Plotly, Pandas
- **PDF Generation**: FPDF
- **Email Service**: SendGrid
- **Web Scraping**: BeautifulSoup, Requests

---

## 🏗️ System Architecture

The system follows a modular design with dedicated components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Document       │────▶│  Analysis       │────▶│  Visualization  │
│  Processing     │     │  Engine         │     │  & Reporting    │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Vector         │     │  Legal          │     │  Email          │
│  Database       │     │  Knowledge Base │     │  Service        │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 🔧 Installation Guide

### Clone the Repository
```sh
git clone https://github.com/yourusername/legal-ai-tool.git
cd legal-ai-tool
```

### Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Set Up API Keys
Create a `.env` file and add your API keys:
```sh
GROQ_API_KEY=your_groq_api_key
SENDGRID_API_KEY=your_sendgrid_api_key
SENDER_EMAIL=your_sender_email
```

### Run the Application
```sh
streamlit run app.py --server.fileWatcherType none
```

---

## 📋 Usage Guide

### Document Analysis
- Upload a legal document (PDF format).
- Click "Analyze Document" to process it.
- View the summary and detailed insights.

### Risk Assessment
- The dashboard categorizes risks as **Critical, High, Medium, or Low**.
- Interactive charts display risk distribution.

### Document Q&A
- Ask specific queries about document contents.
- Uses **RAG (Retrieval Augmented Generation)** for accurate responses.

### Document Comparison
- Upload a second document for comparison.
- View a side-by-side analysis or tabular comparison.
- Generate a comparison report.

### Compliance Analysis
- Detects compliance requirements based on document type.
- Highlights regulatory obligations and recent updates.

---

## 🌟 Implementation Highlights

### Advanced NLP Processing
- Extracts structured content using **PyMuPDF** and **NLTK**.
- Ensures accurate parsing of complex legal documents.

### Semantic Understanding
- Uses **semantic embeddings** instead of simple keyword matching.
- Delivers more precise document summarization and comparisons.

### AI-Powered Q&A
- Employs **Retrieval Augmented Generation (RAG)**.
- Retrieves relevant sections before generating responses.

### Legal-Specific Analysis
- Custom analyzers for different document types (contracts, GDPR, NDAs, etc.).
- Provides specialized insights tailored to document categories.

---

## 🔮 Future Enhancements
- Multi-document correlation for deeper insights.
- Integration with legal case databases.
- Team collaboration and annotation features.
- Support for additional formats (DOCX, HTML, TXT).
- Mobile-friendly version for on-the-go access.
- Fine-tuning of AI models for improved accuracy.

---

## ⚠️ Troubleshooting

### Streamlit & PyTorch Issues
If you encounter a `RuntimeError: Tried to instantiate class 'path._path'`, try:

```python
import os
os.environ["PYTORCH_JIT"] = "0"  # Disable PyTorch JIT
```
Or run Streamlit with:
```sh
streamlit run app.py --server.fileWatcherType none
```

### Missing NLTK Resources
Ensure NLTK resources are properly downloaded:
```python
import nltk
nltk.download(['punkt', 'averaged_perceptron_tagger', 'vader_lexicon'])
```

---

## 📜 License
© 2025 LegalAI - All Rights Reserved. This software is proprietary and confidential.

