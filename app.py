import plotly.express as px
import nltk
nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'vader_lexicon'])
import streamlit as st
import os
import faiss
import numpy as np
import fitz  # PyMuPDF
import pandas as pd
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import Tuple, List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import difflib
import requests
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Initialize NLP resources
nltk.download(['punkt', 'averaged_perceptron_tagger', 'vader_lexicon'])
from nltk.sentiment import SentimentIntensityAnalyzer

# Load environment variables
load_dotenv()

# Constants
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3-70b-8192"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3 

# Email configuration
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# Initialize session state with proper structure
def initialize_session_state():
    session_defaults = {
        'chat_history': [],
        'document_processed': False,
        'text_chunks': [],
        'faiss_index': None,
        'embeddings': None,
        'full_text': "",
        'comparison_result': None,
        'summaries': {},
        'risk_data': {
            'categories': {},
            'total_risks': 0,
            'severity_counts': {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
            'total_score': 0
        }
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Load models with error handling
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        st.error(f"Failed to load embedding model: {str(e)}")
        return None

@st.cache_resource
def load_llm():
    try:
        return ChatGroq(
            model_name=LLM_MODEL_NAME, 
            api_key=os.getenv("GROQ_API_KEY"),
            request_timeout=30
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None

embedding_model = load_embedding_model()
llm = load_llm()
sia = SentimentIntensityAnalyzer()

def extract_text_from_pdf(pdf_file: BytesIO) -> str:
    """Extracts text from PDF documents using PyMuPDF with error handling"""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return ""

def chunk_text(text: str) -> List[str]:
    """Splits text into meaningful chunks with overlap"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    return splitter.split_text(text)

def create_faiss_index(text_chunks: List[str]) -> Tuple[faiss.Index, np.ndarray]:
    """Creates and returns FAISS index with embeddings"""
    if not text_chunks:
        return None, None
        
    try:
        embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype=np.float32))
        return index, embeddings
    except Exception as e:
        st.error(f"Index creation failed: {str(e)}")
        return None, None

def retrieve_relevant_chunks(query: str, index: faiss.Index, text_chunks: List[str]) -> List[str]:
    """Retrieves the top K relevant document chunks from FAISS index"""
    if not index or not text_chunks:
        return []
        
    try:
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        _, indices = index.search(np.array(query_embedding, dtype=np.float32), TOP_K)
        return [text_chunks[i] for i in indices[0] if i < len(text_chunks)]
    except Exception as e:
        st.error(f"Retrieval failed: {str(e)}")
        return []

def generate_rag_response(query: str) -> str:
    """Generates response using Retrieval Augmented Generation"""
    if not st.session_state.faiss_index:
        return "No document processed yet"

    try:
        relevant_chunks = retrieve_relevant_chunks(
            query, 
            st.session_state.faiss_index, 
            st.session_state.text_chunks
        )
        context = "\n\n".join(relevant_chunks)

        prompt_template = PromptTemplate.from_template("""
        Context: {context}
        Question: {query}
        Answer: 
        """)

        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        return llm_chain.run({"context": context, "query": query})
    except Exception as e:
        return f"Error generating response: {str(e)}"

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
        risk_results["total_score"] = min(100, 
            sum([v["score"] for v in risk_results["categories"].values()]) +
            (1 - sentiment['compound']) * 25 +
            min(30, avg_sentence_length * 0.5)
        )
        
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

def generate_summary(text: str) -> str:
    """Robust summary generation with error handling"""
    if not text:
        return "No content to summarize"

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        docs = text_splitter.create_documents([text])
        
        map_template = """Summarize this legal document chunk:
        {docs}
        CONCISE SUMMARY:"""
        map_prompt = PromptTemplate.from_template(map_template)
        
        reduce_template = """Combine these summaries:
        {doc_summaries}
        FINAL SUMMARY:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        
        map_chain = LLMChain(llm=llm, prompt=map_prompt)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
        
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name="doc_summaries"
        )
        
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000
        )
        
        return MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs"
        ).run(docs)
    except Exception as e:
        return f"Summary generation failed: {str(e)}"

def compare_documents(doc1: str, doc2: str) -> str:
    """Document comparison with error handling"""
    try:
        d = difflib.Differ()
        diff = list(d.compare(doc1.splitlines(), doc2.splitlines()))
        
        result = []
        for line in diff:
            if line.startswith('- '):
                result.append(f'<span style="color:red">{line}</span>')
            elif line.startswith('+ '):
                result.append(f'<span style="color:green">{line}</span>')
            else:
                result.append(line)
        
        return '<br>'.join(result)
    except Exception as e:
        return f"Comparison failed: {str(e)}"

def fetch_compliance_guidelines():
    """Web scraper with improved error handling"""
    sources = {
        "GDPR": "https://gdpr-info.eu/",
        "HIPAA": "https://www.hhs.gov/hipaa/for-professionals/index.html"
    }
    
    guidelines = {}
    for name, url in sources.items():
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if name == "GDPR":
                content = soup.find_all('article', limit=5)
            elif name == "HIPAA":
                content = soup.find('div', {'class': 'content'}).find_all('p', limit=5)
            
            guidelines[name] = "\n".join([p.get_text(strip=True) for p in content if p.get_text(strip=True)])
        except Exception as e:
            guidelines[name] = f"Error retrieving {name} guidelines: {str(e)}"
    
    return guidelines

def send_email(to_email: str, subject: str, content: str) -> str:
    """Email sending with comprehensive error handling"""
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD]):
        return "Email configuration incomplete"
        
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(content, 'plain'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, to_email, msg.as_string())
        return "Email sent successfully"
    except Exception as e:
        return f"Email failed: {str(e)}"

def main():
    st.title("📜 Legal Document Analyzer PRO")
    
    # Document Processing Section
    with st.expander("📂 Document Upload & Processing", expanded=True):
        uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"])
        if uploaded_file and not st.session_state.document_processed:
            with st.spinner("Processing document..."):
                try:
                    # Extract text
                    st.session_state.full_text = extract_text_from_pdf(uploaded_file)
                    if not st.session_state.full_text:
                        raise ValueError("Empty document content")
                    
                    # Process document
                    st.session_state.text_chunks = chunk_text(st.session_state.full_text)
                    st.session_state.faiss_index, _ = create_faiss_index(st.session_state.text_chunks)
                    st.session_state.summaries['document'] = generate_summary(st.session_state.full_text)
                    st.session_state.risk_data = advanced_risk_assessment(st.session_state.full_text)
                    st.session_state.document_processed = True
                    st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    st.session_state.document_processed = False

    if st.session_state.document_processed:
        # Chat Interface
        query = st.chat_input("💬 Ask about the document")
        if query:
            with st.spinner("Analyzing..."):
                response = generate_rag_response(query)
                st.session_state.chat_history.extend([
                    ("user", query),
                    ("assistant", response)
                ])
            
            for role, msg in st.session_state.chat_history:
                st.chat_message(role).write(msg)

        # Risk Analysis Section
        with st.expander("📊 Risk Analysis", expanded=True):
            risk_data = st.session_state.risk_data
            
            # Metrics Row
            cols = st.columns(4)
            cols[0].metric("Total Score", f"{risk_data.get('total_score', 0)}/100")
            cols[1].metric("Total Risks", risk_data.get('total_risks', 0))
            cols[2].metric("Critical", risk_data['severity_counts'].get('Critical', 0))
            cols[3].metric("High Risk", risk_data['severity_counts'].get('High', 0))
            
            # Visualizations
            fig1, fig2 = visualize_risks(risk_data)
            if fig1 and fig2:
                st.plotly_chart(fig1, use_container_width=True)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Detailed Breakdown
        with st.expander("🔍 Detailed Analysis"):
            if risk_data.get('categories'):
                df = pd.DataFrame.from_dict(risk_data['categories'], orient='index')
                st.dataframe(df)
            else:
                st.warning("No risk data available")

        # Compliance Section
        with st.expander("📜 Compliance Guidelines"):
            guidelines = fetch_compliance_guidelines()
            for name, text in guidelines.items():
                st.subheader(name)
                st.markdown(f"""<div style='background:#f8f9fa; padding:15px; border-radius:8px'>
                    {text}
                </div>""", unsafe_allow_html=True)
                st.download_button(
                    label=f"Download {name}",
                    data=text,
                    file_name=f"{name}_guidelines.txt"
                )

        # Document Comparison
        with st.expander("🔀 Compare Documents"):
            compare_file = st.file_uploader("Upload comparison PDF", type=["pdf"])
            if compare_file:
                try:
                    compare_text = extract_text_from_pdf(compare_file)
                    comparison = compare_documents(st.session_state.full_text, compare_text)
                    st.markdown(comparison, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Comparison failed: {str(e)}")

        # Summary Section
        with st.expander("📝 Document Summary"):
            summary = st.session_state.summaries.get('document', "No summary available")
            st.write(summary)
            st.download_button(
                "Download Summary",
                data=summary,
                file_name="document_summary.txt"
            )

        # Email Section
        with st.expander("📧 Email Report"):
            email = st.text_input("Recipient Email")
            if st.button("Send Summary"):
                if email:
                    result = send_email(
                        email,
                        "Legal Document Analysis Report",
                        f"Document Summary:\n{summary}\n\nRisk Score: {risk_data.get('total_score', 0)}"
                    )
                    if "success" in result.lower():
                        st.success("Email sent!")
                    else:
                        st.error(result)
                else:
                    st.warning("Please enter an email address")

    else:
        st.info("⬆️ Upload a PDF document to begin analysis")

if __name__ == "__main__":
    main()