import streamlit as st
from modules.economic_report import run_economic_report
from modules.faq_chatbot import run_chatbot
from modules.news_summarizer import run_news_summarizer
from modules.fraud_detection import run_fraud_detection   # ✅ Newly added

# -------------------------------------------------------------
# ⚙️ PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Finance AI Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# 🎨 DARK THEME + ANIMATIONS
# -------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #0F172A 0%, #1E293B 80%);
    color: #E2E8F0;
    animation: fadeIn 0.8s ease-in-out;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #1E293B);
}
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #60A5FA;
    font-family: 'Segoe UI Semibold', sans-serif;
    letter-spacing: 0.5px;
}

/* Text */
p, label, span, div {
    color: #E2E8F0 !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Buttons */
div.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background-color: #1D4ED8;
    color: #FFF;
    transform: scale(1.04);
}

/* Alerts */
div[data-testid="stAlert"] {
    background-color: #1E293B !important;
    border-left: 4px solid #3B82F6 !important;
    color: #E2E8F0 !important;
    border-radius: 8px;
}

/* Sidebar hover */
div[role="radiogroup"] label:hover {
    color: #60A5FA !important;
    transform: scale(1.02);
}

/* Animated content card */
.module-container {
    background: rgba(30, 41, 59, 0.6);
    border-radius: 14px;
    padding: 25px;
    margin-top: 10px;
    box-shadow: 0 0 20px rgba(59,130,246,0.15);
    animation: slideUp 0.8s ease-in-out;
}

/* Animations */
@keyframes fadeIn {
    0% {opacity: 0;}
    100% {opacity: 1;}
}
@keyframes slideUp {
    0% {opacity: 0; transform: translateY(15px);}
    100% {opacity: 1; transform: translateY(0);}
}

/* Footer */
.footer {
    text-align: center;
    color: #94A3B8;
    margin-top: 2rem;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 🧠 SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/679/679720.png", width=70)
st.sidebar.title("🧠 Finance AI Suite")

choice = st.sidebar.radio(
    "📂 Choose a Module",
    (
        "🏦 About App",
        "💹 Economic Report Generator",
        "🕵️‍♂️ Fraud Detection Assistant",
        "🤖 Finance FAQ Chatbot",
        "📰 News Summarizer",
    ),
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("🧑‍💻 **Developed by [Sohan Ghosh](#)**", unsafe_allow_html=True)
st.sidebar.markdown("📊 *M.Sc. in Data Science & AI, University of Calcutta*")

# -------------------------------------------------------------
# 🏠 HEADER
# -------------------------------------------------------------
st.title("💼 Finance AI Suite")
st.caption("All your **LLM + RAG + Finance Analytics** tools in one sleek workspace.")

# -------------------------------------------------------------
# 🔀 ROUTING WITH ANIMATED CONTAINERS
# -------------------------------------------------------------
if choice == "🏦 About App":
    st.markdown('<div class="module-container">', unsafe_allow_html=True)
    st.subheader("⚙️ About Finance AI Suite")
    st.markdown("""
    **Finance AI Suite** integrates **AI-powered analytics**, **retrieval-augmented intelligence**,  
    and **financial data pipelines** — built for advanced market insight generation.

    ### 🧩 Modules Included:
    - 💬 **Finance FAQ Chatbot** → Retrieval-Augmented chatbot over 1,700+ bank FAQs  
    - 💹 **Economic Report Generator** → Global market index analysis & comparisons  
    - 🕵️‍♂️ **Fraud Detection Assistant** → Detect anomalies in transaction data using AI  
    - 📰 **News Summarizer** → Media tone and sentiment analysis with summaries  

    ### 🧠 Core Technologies:
    - Streamlit UI • LangChain • Hugging Face Transformers  
    - ChromaDB • Yahoo Finance API • Google News RSS  

    The suite is designed to bring intelligence, automation, and clarity to modern financial research.
    """)
    st.success("✅ All systems ready — choose a module from the sidebar to begin.")
    st.markdown('</div>', unsafe_allow_html=True)

elif choice == "💹 Economic Report Generator":
    st.markdown('<div class="module-container">', unsafe_allow_html=True)
    st.header("💹 Economic Report Generator")
    st.caption("Analyze market trends, volatility, and cross-country correlations.")
    run_economic_report()
    st.markdown('</div>', unsafe_allow_html=True)

elif choice == "🕵️‍♂️ Fraud Detection Assistant":
    st.markdown('<div class="module-container">', unsafe_allow_html=True)
    st.header("🕵️‍♂️ Financial Fraud Detection Assistant")
    st.caption("Upload transaction data, detect anomalies, and understand suspicious patterns using ML.")
    run_fraud_detection()   # ✅ Call your actual module function
    st.markdown('</div>', unsafe_allow_html=True)

elif choice == "🤖 Finance FAQ Chatbot":
    st.markdown('<div class="module-container">', unsafe_allow_html=True)
    st.header("🤖 Finance FAQ Chatbot (RAG-based)")
    st.caption("Ask financial and banking-related questions — powered by LangChain + Chroma + T5.")
    run_chatbot()
    st.markdown('</div>', unsafe_allow_html=True)

elif choice == "📰 News Summarizer":
    st.markdown('<div class="module-container">', unsafe_allow_html=True)
    st.header("📰 Company News Summarizer & Media Bias Analyzer")
    st.caption("Compare company sentiment and tone through AI summarization and news analysis.")
    run_news_summarizer()
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 📄 FOOTER
# -------------------------------------------------------------
st.markdown("""
<div class="footer">
    <hr>
    <p>© 2025 <b>Finance AI Suite</b> — Developed by <b>Sohan Ghosh</b></p>
</div>
""", unsafe_allow_html=True)
