import plotly.express as px
import nltk
nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'vader_lexicon'])
import streamlit as st
import os
import base64
import faiss
import numpy as np
import fitz  # PyMuPDF
import pandas as pd
from fpdf import FPDF
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
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64

st.set_page_config(
    page_title="LegalDoc Analyst",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="auto"
)

# Initialize NLP resources
nltk.download(['punkt', 'averaged_perceptron_tagger', 'vader_lexicon'])
from nltk.sentiment import SentimentIntensityAnalyzer

# Load environment variables
load_dotenv()
sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
sender_email = os.getenv("SENDER_EMAIL")

if not sendgrid_api_key or not sender_email:
    raise ValueError("Missing SendGrid API Key or Sender Email in environment variables.")

# Constants
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3-70b-8192"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3 



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
        risk_results["total_score"] = round(min(100, 
            sum([v["score"] for v in risk_results["categories"].values()]) +
            (1 - sentiment['compound']) * 25 +
            min(30, avg_sentence_length * 0.5)
        ),)
        
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
    """Enhanced compliance checklist with structured requirements"""
    checklists = {
        "GDPR": {
            "checklist": [
                "🛡️ Lawful basis for data processing documented",
                "📝 Clear privacy notice provided to data subjects",
                "🔒 Data minimization practices implemented",
                "⏱️ Right to erasure procedure established",
                "📤 Data portability mechanism available",
                "🕵️ Data Protection Impact Assessments conducted",
                "📞 Designated Data Protection Officer (if required)",
                "⚠️ 72-hour breach notification process in place"
            ],
            "reference": "https://gdpr-info.eu/"
        },
        "HIPAA": {
            "checklist": [
                "🏥 Patient authorization for PHI disclosure",
                "📁 Minimum Necessary Standard implemented",
                "🔐 Physical and technical safeguards for ePHI",
                "📝 Notice of Privacy Practices displayed",
                "👥 Workforce security training conducted",
                "📅 6-year documentation retention policy",
                "🚨 Breach notification protocol established",
                "📊 Business Associate Agreements in place"
            ],
            "reference": "https://www.hhs.gov/hipaa/"
        }
    }

    try:
        # Add live updates from official sources
        for name in checklists.keys():
            try:
                response = requests.get(
                    checklists[name]["reference"],
                    headers={'User-Agent': 'Mozilla/5.0'},
                    timeout=5
                )
                soup = BeautifulSoup(response.text, 'html.parser')
                
                updates = []
                if name == "GDPR":
                    articles = soup.find_all('article', limit=3)
                    updates = [a.get_text(strip=True) for a in articles if a.get_text(strip=True)]
                elif name == "HIPAA":
                    content = soup.find('div', {'class': 'content'})
                    updates = [p.get_text(strip=True) for p in content.find_all('p', limit=3)] if content else []
                
                checklists[name]["latest_updates"] = updates
            
            except Exception as e:
                checklists[name]["latest_updates"] = [f"⚠️ Failed to retrieve live updates: {str(e)}"]

        return checklists

    except Exception as e:
        return {"error": f"Compliance system unavailable: {str(e)}"}

def generate_pdf(summary, risk_data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Legal Document Analysis Report", ln=True, align="C")
    pdf.ln(10)  # Line break

    pdf.multi_cell(0, 10, f"Document Summary:\n{summary}")
    pdf.ln(5)

    pdf.multi_cell(0, 10, f"Risk Score: {risk_data.get('total_score', 0)}")

    # Save PDF as a string
    pdf_data = pdf.output(dest="S").encode("latin1")  # Generate PDF as a string

    # Convert to BytesIO
    pdf_buffer = BytesIO(pdf_data)

    return pdf_buffer

def send_email(recipient_email, pdf_buffer):
    sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
    sender_email = os.getenv("SENDER_EMAIL")

    if not sendgrid_api_key or not sender_email:
        st.error("⚠ Missing SendGrid API Key or Sender Email in environment variables.")
        return False

    # Read and encode PDF
    pdf_buffer.seek(0)
    encoded_pdf = base64.b64encode(pdf_buffer.read()).decode()

    # Construct email
    message = Mail(
        from_email=sender_email,
        to_emails=recipient_email,
        subject="📄 Legal Document Report",
        html_content="Please find the attached legal document report."
    )

    # Attach PDF
    attachment = Attachment(
        FileContent(encoded_pdf),
        FileName("Legal_Report.pdf"),
        FileType("application/pdf"),
        Disposition("attachment")
    )
    message.attachment = attachment

    try:
        sg = SendGridAPIClient(sendgrid_api_key)
        sg.send(message)
        st.success("✅ Email sent successfully!")
        return True
    except Exception as e:
        st.error(f"⚠ Email sending failed: {str(e)}")
        return False

def main():
    
    
    # Custom CSS styling
    st.markdown("""
    <style>
        .main {background-color: #f5f7fb;}
        .stButton>button {border-radius: 8px; padding: 0.5rem 1rem;}
        .stDownloadButton>button {width: 100%;}
        .stExpander .st-emotion-cache-1hynsf2 {border-radius: 10px;}
        .metric-box {padding: 20px; border-radius: 10px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
        .risk-critical {color: #dc3545!important;}
        .risk-high {color: #ff6b6b!important;}
        .risk-medium {color: #ffd93d!important;}
        .risk-low {color: #6c757d!important;}
                
    </style>
    """, unsafe_allow_html=True)

    # App Header
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
    with col2:
        st.title("Legal Document Summarizer and Analyzer")
        st.markdown("**AI-powered Contract Analysis & Risk Assessment**")

    # Main Layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Document Analysis", 
        "📊 Risk Dashboard", 
        "🔀 Comparison", 
        "📜 Compliance", 
        "📧 Report"
    ])

    # Document Processing Section
    with tab1:
        st.header("Document Processing")
        with st.container(border=True):
            uploaded_file = st.file_uploader("Upload Legal Document (PDF)", type=["pdf"])
            if uploaded_file and not st.session_state.document_processed:
                if st.button("Analyze Document", type="primary"):
                    with st.status("Processing document...", expanded=True) as status:
                        try:
                            st.write("Extracting text...")
                            st.session_state.full_text = extract_text_from_pdf(uploaded_file)
                            
                            st.write("Chunking text...")
                            st.session_state.text_chunks = chunk_text(st.session_state.full_text)
                            
                            st.write("Creating search index...")
                            st.session_state.faiss_index, _ = create_faiss_index(st.session_state.text_chunks)
                            
                            st.write("Generating summary...")
                            st.session_state.summaries['document'] = generate_summary(st.session_state.full_text)
                            
                            st.write("Assessing risks...")
                            st.session_state.risk_data = advanced_risk_assessment(st.session_state.full_text)
                            
                            status.update(label="Analysis Complete!", state="complete", expanded=False)
                            st.session_state.document_processed = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Processing failed: {str(e)}")
                            st.session_state.document_processed = False

        if st.session_state.document_processed:
            with st.container(border=True):
                st.subheader("Document Summary")
                st.write(st.session_state.summaries.get('document', "No summary available"))
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.download_button(
                        "Download Text Summary",
                        data=st.session_state.summaries.get('document', ""),
                        file_name="document_summary.txt",
                        use_container_width=True
                    )
                with col_d2:
                    if st.button("Generate Full Report PDF", use_container_width=True):
                        with st.spinner("Generating PDF..."):
                            pdf_buffer = generate_pdf(
                                st.session_state.summaries['document'],
                                st.session_state.risk_data
                            )
                            st.session_state.pdf_buffer = pdf_buffer
                            st.success("PDF ready for download!")

    # Risk Dashboard
    
    if st.session_state.document_processed:
        with tab2:
            st.header("Risk Analysis Dashboard")
            risk_data = st.session_state.risk_data
        
        # Risk Metrics
            with st.container(border=True):
                cols = st.columns(4)
                metric_style = """
                    <style>
                        .metric-box {
                            padding: 20px;
                            border-radius: 10px;
                            background: white;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            height: 150px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                        }
                        .metric-title {
                            font-size: 1.1rem;
                            margin-bottom: 8px;
                            font-weight: 600;
                            color: #666;
                        }
                        .metric-value {
                            font-size: 2.5rem;
                            font-weight: 700;
                            line-height: 1.2;
                            color: #dc3545 !important;
                        }
                        .metric-subtext {
                            font-size: 1rem;
                            color: #666;
                        }
                        .risk-critical { color: #dc3545; }
                        .risk-high { color: #ff6b6b; }
                    </style>
                """
                st.markdown(metric_style, unsafe_allow_html=True)

            # Overall Risk Score
            with cols[0]:
                st.markdown(f'''
                    <div class="metric-box">
                        <div class="metric-title risk-critical">Overall Risk Score</div>
                        <div class="metric-value risk-critical">
                            {risk_data.get("total_score", 0)}
                            <span class="metric-subtext">/100</span>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

            # Total Risks
            with cols[1]:
                st.markdown(f'''
                    <div class="metric-box">
                        <div class="metric-title">Total Risks</div>
                        <div class="metric-value">
                            {risk_data.get("total_risks", 0)}
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

            # Critical Risks
            with cols[2]:
                st.markdown(f'''
                    <div class="metric-box">
                        <div class="metric-title risk-critical">Critical Risks</div>
                        <div class="metric-value risk-critical">
                            {risk_data["severity_counts"].get("Critical", 0)}
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

            # High Risks
            with cols[3]:
                st.markdown(f'''
                    <div class="metric-box">
                        <div class="metric-title risk-high">High Risks</div>
                        <div class="metric-value risk-high">
                            {risk_data["severity_counts"].get("High", 0)}
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

        # Rest of the dashboard...

            # Visualizations
            with st.container(border=True):
                fig1, fig2 = visualize_risks(risk_data)
                if fig1 and fig2:
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.plotly_chart(fig1, use_container_width=True)
                    with col_v2:
                        st.plotly_chart(fig2, use_container_width=True)

            # Detailed Risk Breakdown
            with st.container(border=True):
                st.subheader("Risk Category Breakdown")
                if risk_data.get('categories'):
                    df = pd.DataFrame.from_dict(risk_data['categories'], orient='index')
                    st.dataframe(
                        df,
                        column_config={
                            "score": st.column_config.ProgressColumn(
                                "Score",
                                help="Risk score (0-40)",
                                format="%f",
                                min_value=0,
                                max_value=40,
                            )
                        },
                        use_container_width=True
                    )

    # Document Comparison
    with tab3:
        st.header("Document Comparison")
        if st.session_state.document_processed:
            with st.container(border=True):
                compare_file = st.file_uploader("Upload Comparison Document", type=["pdf"])
                if compare_file:
                    try:
                        compare_text = extract_text_from_pdf(compare_file)
                        comparison = compare_documents(st.session_state.full_text, compare_text)
                        st.markdown(
                            f'<div style="border:1px solid #eee; padding:20px; border-radius:8px">'
                            f'{comparison}</div>',
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(f"Comparison failed: {str(e)}")

    # Compliance Section
    with tab4:
        st.header("Compliance Checklists")
        guidelines = fetch_compliance_guidelines()
    
        if isinstance(guidelines, dict) and "error" not in guidelines:
            for regulation, data in guidelines.items():
                with st.expander(f"🔍 {regulation} Compliance Checklist"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader(f"{regulation} Requirements")
                        for item in data.get("checklist", []):
                            st.markdown(f"- {item}")
                    
                    with col2:
                        st.download_button(
                            label=f"Download {regulation} Checklist",
                            data="\n".join(data.get("checklist", [])),
                            file_name=f"{regulation}_checklist.txt",
                            use_container_width=True
                        )
                
                    if data.get("latest_updates"):
                        st.markdown("---")
                        st.caption(f"**Latest {regulation} Updates**")
                        for update in data["latest_updates"][:3]:
                            st.markdown(f"📢 {update}")
        else:
            st.error("Failed to load compliance guidelines")

    # Report Section
    with tab5:
        st.header("Report Generation")
        if st.session_state.document_processed:
            with st.container(border=True):
                st.subheader("Email Report")
                email = st.text_input("Recipient Email Address", placeholder="legal@company.com")
                
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    if st.button("📧 Send Email Report", use_container_width=True):
                        if email and hasattr(st.session_state, 'pdf_buffer'):
                            if send_email(email, st.session_state.pdf_buffer):
                                st.success("Report sent successfully!")
                            else:
                                st.error("Failed to send email")
                        else:
                            st.warning("Please generate PDF first and enter valid email")
                
                with col_e2:
                    if hasattr(st.session_state, 'pdf_buffer'):
                        st.download_button(
                            label="⬇️ Download Full Report",
                            data=st.session_state.pdf_buffer,
                            file_name="Legal_Analysis_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

    # Chat Interface (Floating in Sidebar)
    if st.session_state.document_processed:
        with st.sidebar:
            st.header("Document Q&A")
            for role, msg in st.session_state.chat_history[-3:]:
                with st.chat_message(role):
                    st.write(msg)
            
            if query := st.chat_input("Ask about the document..."):
                with st.spinner("Analyzing..."):
                    response = generate_rag_response(query)
                    st.session_state.chat_history.extend([
                        ("user", query),
                        ("assistant", response)
                    ])
                    st.rerun()
            # License Footer
            st.markdown("---")
            st.caption("""
                © 2025 VidzAI - All Rights Reserved  
                This software is proprietary and confidential  
                Unauthorized use or distribution prohibited
            """)        

if __name__ == "__main__":
    main()