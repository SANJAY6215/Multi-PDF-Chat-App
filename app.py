import streamlit as st
from dotenv import load_dotenv
import os
import time

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# ==========================
# Configuration
# ==========================
load_dotenv()

st.set_page_config(
    page_title="Research Agent — AI PDF Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Custom CSS
# ==========================
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.85) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(139, 92, 246, 0.15);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0d4ff !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown span {
    color: #b0a4d4 !important;
}

/* ── Sidebar Upload Area ── */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    border: 2px dashed rgba(139, 92, 246, 0.35) !important;
    border-radius: 16px !important;
    padding: 12px !important;
    background: rgba(139, 92, 246, 0.05) !important;
    transition: all 0.3s ease;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
    border-color: rgba(139, 92, 246, 0.6) !important;
    background: rgba(139, 92, 246, 0.1) !important;
}

/* ── Primary Button ── */
.stButton > button {
    background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 50%, #5b21b6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.45) !important;
    background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 50%, #6d28d9 100%) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Text Input ── */
.stTextInput > div > div > input {
    background: rgba(30, 27, 60, 0.7) !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 14px !important;
    color: #e8e0ff !important;
    padding: 14px 20px !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus {
    border-color: rgba(139, 92, 246, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
}

.stTextInput > div > div > input::placeholder {
    color: rgba(176, 164, 212, 0.5) !important;
}

/* ── Chat Messages ── */
.chat-message-user {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(109, 40, 217, 0.1));
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 18px 18px 4px 18px;
    padding: 16px 22px;
    margin: 10px 0;
    color: #d4c8f0;
    font-size: 0.95rem;
    line-height: 1.6;
    max-width: 85%;
    margin-left: auto;
    animation: fadeSlideUp 0.4s ease-out;
}

.chat-message-bot {
    background: linear-gradient(135deg, rgba(30, 27, 60, 0.8), rgba(40, 36, 80, 0.6));
    border: 1px solid rgba(139, 92, 246, 0.1);
    border-radius: 18px 18px 18px 4px;
    padding: 20px 24px;
    margin: 10px 0;
    color: #e0daf0;
    font-size: 0.95rem;
    line-height: 1.7;
    max-width: 90%;
    animation: fadeSlideUp 0.5s ease-out;
}

/* ── Animations ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.6; }
}

/* ── Hero Header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    animation: fadeSlideUp 0.6s ease-out;
}

.hero-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #c4b5fd, #8b5cf6, #a78bfa, #ddd6fe);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 6s ease infinite;
    margin-bottom: 0.4rem;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    color: #9f92c4;
    font-size: 1.1rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* ── Stat Cards ── */
.stat-card {
    background: rgba(30, 27, 60, 0.6);
    border: 1px solid rgba(139, 92, 246, 0.12);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}

.stat-card:hover {
    border-color: rgba(139, 92, 246, 0.3);
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(139, 92, 246, 0.1);
}

.stat-icon {
    font-size: 1.8rem;
    margin-bottom: 6px;
}

.stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #c4b5fd;
}

.stat-label {
    font-size: 0.78rem;
    color: #8b80b0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* ── Section Divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.3), transparent);
    margin: 1.5rem 0;
    border: none;
}

/* ── Sidebar file list items ── */
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
    background: rgba(139, 92, 246, 0.08) !important;
    border-radius: 10px !important;
    margin: 4px 0 !important;
}

/* ── Success / Warning / Error boxes ── */
.stSuccess {
    background-color: rgba(16, 185, 129, 0.1) !important;
    border: 1px solid rgba(16, 185, 129, 0.25) !important;
    border-radius: 12px !important;
    color: #6ee7b7 !important;
}

.stWarning {
    background-color: rgba(245, 158, 11, 0.1) !important;
    border: 1px solid rgba(245, 158, 11, 0.25) !important;
    border-radius: 12px !important;
}

.stError {
    background-color: rgba(239, 68, 68, 0.1) !important;
    border: 1px solid rgba(239, 68, 68, 0.25) !important;
    border-radius: 12px !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-color: #8b5cf6 !important;
}

/* ── Footer ── */
.custom-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #6b6190;
    font-size: 0.82rem;
    letter-spacing: 0.3px;
}

.custom-footer a {
    color: #8b5cf6;
    text-decoration: none;
}

.footer-tech {
    display: inline-flex;
    gap: 8px;
    align-items: center;
    margin-top: 8px;
    flex-wrap: wrap;
    justify-content: center;
}

.tech-badge {
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    color: #a78bfa;
    font-weight: 500;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(139, 92, 246, 0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(139, 92, 246, 0.5);
}

/* ── Chat container ── */
.chat-container {
    max-height: 60vh;
    overflow-y: auto;
    padding: 10px 0;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #7b6fa5;
    animation: fadeSlideUp 0.8s ease-out;
}

.empty-state-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    animation: pulse 3s ease-in-out infinite;
}

.empty-state h3 {
    color: #a89fd0;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.empty-state p {
    font-size: 0.92rem;
    line-height: 1.6;
}

/* ── Sidebar section headers ── */
.sidebar-section {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(109, 40, 217, 0.04));
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    border: 1px solid rgba(139, 92, 246, 0.08);
}

/* ── Label text ── */
.stTextInput label, .stFileUploader label {
    color: #b0a4d4 !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Session State
# ==========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# ==========================
# Helper Functions
# ==========================

def extract_pdf_text(pdf_files):
    text = ""
    total_pages = 0

    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        total_pages += len(pdf_reader.pages)

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    st.session_state.total_pages = total_pages
    return text


def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)
    st.session_state.total_chunks = len(chunks)
    return chunks


def build_vector_store(chunks):

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vector_store = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


def get_qa_chain():

    prompt_template = """
    You are a helpful, precise research assistant. Answer the question using ONLY 
    the provided context. Be thorough but concise. Use bullet points or numbered 
    lists when appropriate for readability.

    If the answer is not available in the context, say:
    "I couldn't find the answer in the uploaded documents. Try rephrasing your question or uploading additional documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain


def answer_question(question):

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(question)

    chain = get_qa_chain()

    response = chain(
        {
            "input_documents": docs,
            "question": question
        },
        return_only_outputs=True
    )

    return response["output_text"]


# ==========================
# Sidebar
# ==========================

with st.sidebar:

    # ── Logo / Brand ──
    st.markdown("""
    <div style="text-align: center; padding: 1.2rem 0 0.8rem;">
        <span style="font-size: 2.5rem;">🧠</span>
        <h2 style="margin: 0.3rem 0 0; font-size: 1.3rem; font-weight: 700;
                    background: linear-gradient(135deg, #c4b5fd, #8b5cf6);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text;">Research Agent</h2>
        <p style="font-size: 0.78rem; color: #7b6fa5; margin-top: 2px;">
            Powered by Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Upload Section ──
    st.markdown("#### 📄 Upload Documents")
    st.markdown(
        '<p style="font-size: 0.82rem; color: #8b80b0; margin-bottom: 8px;">'
        'Upload your PDF files to get started</p>',
        unsafe_allow_html=True
    )

    pdf_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if pdf_files:
        st.markdown(
            f'<p style="font-size: 0.82rem; color: #a78bfa; margin: 8px 0 4px;">'
            f'📎 {len(pdf_files)} file{"s" if len(pdf_files) != 1 else ""} selected</p>',
            unsafe_allow_html=True
        )

    st.markdown("")  # spacing

    if st.button("⚡ Process Documents", use_container_width=True):

        if not pdf_files:
            st.warning("⚠️ Please upload at least one PDF.")
        else:
            with st.spinner("🔄 Extracting text from PDFs..."):
                raw_text = extract_pdf_text(pdf_files)
                time.sleep(0.3)

            with st.spinner("✂️ Splitting into chunks..."):
                chunks = create_chunks(raw_text)
                time.sleep(0.3)

            with st.spinner("🧬 Building vector embeddings..."):
                build_vector_store(chunks)

            st.session_state.docs_processed = True
            st.session_state.doc_count = len(pdf_files)

            st.success("✅ Documents processed successfully!")
            st.balloons()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Processing Stats ──
    if st.session_state.docs_processed:
        st.markdown("#### 📊 Document Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">📄</div>
                <div class="stat-value">{st.session_state.doc_count}</div>
                <div class="stat-label">Documents</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">📃</div>
                <div class="stat-value">{st.session_state.total_pages}</div>
                <div class="stat-label">Pages</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">🧩</div>
                <div class="stat-value">{st.session_state.total_chunks}</div>
                <div class="stat-label">Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">💬</div>
                <div class="stat-value">{len(st.session_state.chat_history)}</div>
                <div class="stat-label">Questions</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    else:
        # ── Instructions ──
        st.markdown("#### 💡 How to Use")
        st.markdown("""
        <div style="font-size: 0.85rem; color: #8b80b0; line-height: 1.8;">
            <b style="color: #a78bfa;">1.</b> Upload one or more PDF files<br>
            <b style="color: #a78bfa;">2.</b> Click <b style="color:#c4b5fd;">Process Documents</b><br>
            <b style="color: #a78bfa;">3.</b> Ask any question about your PDFs<br>
            <b style="color: #a78bfa;">4.</b> Get AI-powered answers instantly
        </div>
        """, unsafe_allow_html=True)


# ==========================
# Main Content
# ==========================

# ── Hero Header ──
st.markdown("""
<div class="hero-header">
    <h1>🧠 AI Research Agent</h1>
    <p class="hero-subtitle">Upload your documents • Ask questions • Get intelligent answers</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Chat History ──
if st.session_state.chat_history:
    for entry in st.session_state.chat_history:
        st.markdown(
            f'<div class="chat-message-user">🙋 &nbsp; {entry["question"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="chat-message-bot">🤖 &nbsp; {entry["answer"]}</div>',
            unsafe_allow_html=True
        )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

elif not st.session_state.docs_processed:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">📚</div>
        <h3>No documents uploaded yet</h3>
        <p>Upload your PDF documents using the sidebar and click 
        <b>Process Documents</b> to get started.<br>
        Then come back here to ask questions!</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">💬</div>
        <h3>Documents ready!</h3>
        <p>Your documents have been processed. Ask a question below to get started.</p>
    </div>
    """, unsafe_allow_html=True)


# ── Question Input ──
question = st.text_input(
    "Ask a question about your documents",
    placeholder="e.g. What are the key findings in the report?",
    label_visibility="collapsed"
)

if question:

    with st.spinner("🔍 Searching documents and generating answer..."):

        try:

            if not os.path.exists("faiss_index"):
                st.error("📌 Please process documents first using the sidebar.")
            else:
                answer = answer_question(question)

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer
                })

                st.rerun()

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")


# ── Footer ──
st.markdown("""
<div class="custom-footer">
    <div class="section-divider"></div>
    <p>Built with ❤️ using cutting-edge AI</p>
    <div class="footer-tech">
        <span class="tech-badge">Gemini 2.5 Flash</span>
        <span class="tech-badge">FAISS</span>
        <span class="tech-badge">LangChain</span>
        <span class="tech-badge">Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)