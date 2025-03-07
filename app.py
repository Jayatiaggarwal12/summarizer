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

# Initialize session state
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
        'risk_data': None
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Load models
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_llm():
    return ChatGroq(model_name=LLM_MODEL_NAME, api_key=os.getenv("GROQ_API_KEY"))

embedding_model = load_embedding_model()
llm = load_llm()
sia = SentimentIntensityAnalyzer()

def extract_text_from_pdf(pdf_file: BytesIO) -> str:
    """Extracts text from PDF documents using PyMuPDF."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return ""

def chunk_text(text: str) -> List[str]:
    """Splits text into meaningful chunks with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    return splitter.split_text(text)

def create_faiss_index(text_chunks: List[str]) -> Tuple[faiss.Index, np.ndarray]:
    """Creates and returns FAISS index with embeddings."""
    embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, embeddings

def retrieve_relevant_chunks(query: str, index: faiss.Index, text_chunks: List[str], embeddings: np.ndarray) -> List[str]:
    """Retrieves the top K relevant document chunks from FAISS index."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), TOP_K)
    retrieved_chunks = [text_chunks[i] for i in indices[0]]
    return retrieved_chunks

def generate_rag_response(query: str) -> str:
    """Retrieves relevant document chunks and generates a response using RAG."""
    if not st.session_state.faiss_index:
        return "No document has been processed yet."

    relevant_chunks = retrieve_relevant_chunks(query, st.session_state.faiss_index, st.session_state.text_chunks, st.session_state.embeddings)
    context = "\n\n".join(relevant_chunks)

    prompt_template = PromptTemplate.from_template("""
    Given the following legal document context, answer the user's query.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    RESPONSE:
    """)

    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    response = llm_chain.run({"context": context, "query": query})
    
    return response

def advanced_risk_assessment(text: str) -> Dict[str, float]:
    """Advanced NLP-based risk detection using multiple techniques."""
    risk_categories = {
        "Compliance": {"keywords": ["regulation", "legal", "gdpr", "hipaa"], "weight": 1.5},
        "Financial": {"keywords": ["penalty", "fine", "liability"], "weight": 2.0},
        "Operational": {"keywords": ["termination", "breach", "default"], "weight": 1.2}
    }
    
    # Sentiment and complexity analysis
    sentiment = sia.polarity_scores(text)
    sentences = nltk.sent_tokenize(text)
    avg_sentence_length = sum(len(nltk.word_tokenize(s)) for s in sentences) / len(sentences) if sentences else 0
    
    scores = {}
    total_score = 0
    
    for category, config in risk_categories.items():
        count = sum(text.lower().count(keyword) for keyword in config["keywords"])
        scores[category] = min(40, count * config["weight"])
        total_score += scores[category]
    
    # Add NLP-based scores
    scores["Sentiment Risk"] = (1 - sentiment['compound']) * 25
    scores["Complexity Risk"] = min(30, avg_sentence_length * 0.5)
    scores["Total"] = min(100, total_score + scores["Sentiment Risk"] + scores["Complexity Risk"])
    
    return scores

def generate_summary(text: str) -> str:
    """Generate summary using map_reduce method."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    
    docs = text_splitter.create_documents([text])
    
    # Map-Reduce implementation
    map_template = """Summarize this legal document chunk:
    {docs}
    CONCISE SUMMARY:"""
    map_prompt = PromptTemplate.from_template(map_template)
    
    reduce_template = """Combine these summaries into final version:
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

def compare_documents(doc1: str, doc2: str) -> str:
    """Compare two documents with highlighted differences."""
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

def fetch_compliance_guidelines():
    """Fetch compliance guidelines via web scraping."""
    sources = {
        "GDPR": "https://gdpr-info.eu/",
        "HIPAA": "https://www.hhs.gov/hipaa/for-professionals/index.html"
    }
    
    guidelines = {}
    for name, url in sources.items():
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if name == "GDPR":
                content = soup.find_all('article', limit=5)
            elif name == "HIPAA":
                content = soup.find('div', {'class': 'content'}).find_all('p', limit=5)
            
            guidelines[name] = "\n".join([p.get_text() for p in content])
        except Exception as e:
            guidelines[name] = f"Error retrieving {name} guidelines: {str(e)}"
    
    return guidelines

def main():
    st.title(" Advanced Legal Document Summarizer")
    
    # Document Processing Section
    with st.expander(" Document Upload & Processing"):
        uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"])
        if uploaded_file and not st.session_state.document_processed:
            with st.spinner("Processing document..."):
                try:
                    st.session_state.full_text = extract_text_from_pdf(uploaded_file)
                    st.session_state.text_chunks = chunk_text(st.session_state.full_text)
                    st.session_state.faiss_index, _ = create_faiss_index(st.session_state.text_chunks)
                    st.session_state.document_processed = True
                    st.session_state.summaries['document'] = generate_summary(st.session_state.full_text)
                    st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

    if st.session_state.document_processed:
        # Chat interface
        query = st.chat_input(" Ask about the document")
        if query:
            with st.spinner(" Generating response..."):
                response = generate_rag_response(query)
                st.session_state.chat_history.append(("user", query))
                st.session_state.chat_history.append(("assistant", response))
            
            for role, message in st.session_state.chat_history:
                st.chat_message(role).write(message)
                
        # Risk Analysis & Visualization
        with st.expander(" Risk Assessment"):
            risk_data = advanced_risk_assessment(st.session_state.full_text)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Total Risk Score", f"{risk_data['Total']}/100")
                st.download_button(
                    label=" Download Risk Report",
                    data=pd.DataFrame.from_dict(risk_data, orient='index').to_csv(),
                    file_name="risk_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                chart_data = pd.DataFrame({
                    'Category': list(risk_data.keys()),
                    'Score': list(risk_data.values())
                }).iloc[:-1]  # Exclude total score
                st.bar_chart(chart_data.set_index('Category'))

        # Document Comparison
        with st.expander(" Document Comparison"):
            compare_file = st.file_uploader("Upload comparison PDF", type=["pdf"])
            if compare_file:
                compare_text = extract_text_from_pdf(compare_file)
                st.session_state.comparison_result = compare_documents(
                    st.session_state.full_text, compare_text
                )
                st.markdown(st.session_state.comparison_result, unsafe_allow_html=True)

        

        # Document Summary
        with st.expander(" Document Summary"):
            st.write(st.session_state.summaries['document'])
            st.download_button(
                label=" Download Summary",
                data=st.session_state.summaries['document'],
                file_name="document_summary.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()