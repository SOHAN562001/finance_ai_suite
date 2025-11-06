# 🧠 Finance AI Suite  
### _An Integrated LLM + RAG–Powered Financial Analytics Platform_  
**Developed by [Sohan Ghosh](#)**  
_M.Sc. in Data Science & Artificial Intelligence, University of Calcutta_

---

## 🚀 Overview

**Finance AI Suite** is a unified **AI-powered financial analytics system** that brings together **Large Language Models (LLMs)**, **Retrieval-Augmented Generation (RAG)**, and **Machine Learning** to deliver intelligent, explainable insights for finance and banking.

This project shows how modern AI frameworks like **LangChain**, **ChromaDB**, and **Hugging Face Transformers** can be integrated into one cohesive, production-ready Streamlit platform for **data-driven decision support**.

---

## 🧩 Modules Included

| Module | Description | Technologies Used |
|--------|--------------|-------------------|
| 💬 **Finance FAQ Chatbot** | RAG-based chatbot answering 1,700+ finance & banking FAQs using vector similarity and LLM reasoning. | LangChain, ChromaDB, MiniLM embeddings, T5 |
| 💹 **Economic Report Generator** | Automated generation of financial market reports, volatility metrics, and AI-interpreted insights. | yFinance, Pandas, Plotly, LangChain |
| 🕵️‍♂️ **Fraud Detection Assistant** | Detects anomalies in financial transactions using unsupervised ML. | IsolationForest, scikit-learn, Streamlit |
| 📰 **News Summarizer & Bias Analyzer** | Fetches financial news, performs sentiment analysis, and compares company-level tone. | BeautifulSoup, VADER, Requests |

---

## 🧠 Project Highlights

### 🧩 Finance FAQ Chatbot (RAG)
- Uses **Chroma Vector Database** for 1,764 cleaned banking FAQs.  
- Embeds text with **Hugging Face Sentence Transformer** (`all-MiniLM-L6-v2`).  
- Performs **semantic retrieval** and **context-aware generation** using FLAN-T5.  
- Implements **LangChain Retrieval Pipeline** for contextual answers.  

📘 _Example_  
> Q: What is NEFT?  
> A: NEFT is an electronic fund transfer system operating on a deferred net settlement basis between banks in India.

---

### 📈 Economic Report Generator
- Fetches **index data** (NIFTY 50, S&P 500, NASDAQ, etc.) via **Yahoo Finance**.  
- Generates **line charts**, **correlation heatmaps**, and **volatility stats**.  
- Compares global indices (e.g., India vs USA) and computes correlations.  
- Summarizes findings with **LLM-driven insights**.  

📊 _Example Insight_  
> Between Aug–Nov 2025, NIFTY 50 rose 4.1%, correlating 0.77 with S&P 500 — showing strong global co-movement.

---

### 🔍 Fraud Detection Assistant
- Upload any **CSV** of transactions (Amount, Merchant, Time, Location).  
- Runs **IsolationForest** for unsupervised anomaly detection.  
- Flags suspicious transactions and computes **Fraud Severity (0-100)**.  
- Summarizes **Customer Risk Index** with visual dashboards and AI explanations.  

🧾 _Example Output_  
> 5.6 % of transactions flagged anomalous, largely due to late-night high-value spikes.

---

### 🗞️ News Summarizer & Media Bias Analyzer
- Retrieves company-specific news via **Yahoo Finance API** and **Google News RSS**.  
- Cleans text with **BeautifulSoup**, analyzes tone using **VADER Sentiment**.  
- Visualizes bias per publisher and generates comparative summaries via LLM.  

📰 _Example Insight_  
> Reliance Industries saw 61 % positive coverage; Tata Motors 44 % neutral — reflecting cautious sentiment.

---

## ⚙️ Technology Stack

| Layer | Tools / Libraries |
|-------|-------------------|
| **Frontend UI** | Streamlit + Custom CSS (dark theme, animations) |
| **Backend AI/NLP** | LangChain, Transformers (FLAN-T5) |
| **Vector Database** | ChromaDB |
| **Data APIs** | Yahoo Finance (`yfinance`), Google News RSS |
| **ML / Analytics** | scikit-learn, Pandas, NumPy |
| **Visualization** | Matplotlib, Plotly |
| **Sentiment** | VADER (NLTK) |
| **Deployment** | Streamlit Cloud / AWS EC2 |

---

## 🧬 System Architecture

```text
User Interface (Streamlit)
│
├── [Module 1] Finance FAQ Chatbot
│       └── LangChain + ChromaDB + LLM
│
├── [Module 2] Economic Report Generator
│       └── yFinance + LLM summary
│
├── [Module 3] Fraud Detection Assistant
│       └── IsolationForest + Severity/Risk Index
│
└── [Module 4] News Summarizer
        └── Yahoo/Google News + VADER Sentiment
📂 Folder Structure
bash
Copy code
finance_ai_suite/
│
├── app.py                        # Main Streamlit entry
├── requirements.txt              # All dependencies
│
├── modules/
│   ├── faq_chatbot.py            # RAG Chatbot
│   ├── economic_report.py        # Market Analysis
│   ├── fraud_detection.py        # Anomaly Detection
│   ├── news_summarizer.py        # Sentiment & Bias Module
│   └── utils.py                  # Helper Utilities
│
├── data/
│   ├── bank_faqs.csv
│   ├── chroma_bank_faqs/
│   └── fraud_transactions_realistic.csv
│
└── README.md
🧾 Installation & Setup
1️⃣ Clone Repository
bash
Copy code
git clone https://github.com/yourusername/finance_ai_suite.git
cd finance_ai_suite
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run Streamlit App
bash
Copy code
streamlit run app.py
4️⃣ Open in Browser
http://localhost:8501

📸 Screenshots (You Will Add)
Add your screenshots here 👇

🏠 Home Dashboard

💬 Chatbot Response

📈 Economic Report Graph

🕵️ Fraud Detection Visualization

🗞️ News Sentiment Analysis

🔍 Customer Risk Chart

📊 Correlation Heatmap

💡 AI Summary Cards

🎯 Severity Distribution Plot

📂 Data Upload Interface

📊 Evaluation Summary
Module	Status	Metric	
FAQ Chatbot	 Done	90 % Context Precision	
Economic Report	 Done	r > 0.75 Correlation	
News Summarizer	 Done	~85 % Sentiment Accuracy	
Fraud Detection	 Done	IsolationForest (unsupervised)	

🧩 Challenges & Solutions
Challenge	Solution
API Rate Limit (Yahoo Finance)	Fallback to Google News RSS
Chroma Cache Issues	Auto-clear with unique session IDs
Visualization Lag	Used st.cache_data() for speed
LLM Memory Load	Used quantized FLAN-T5-small

🔮 Future Enhancements
🔊 Voice-enabled Chatbot (Whisper + TTS)

📈 Portfolio Risk Analyzer (Sharpe, VaR)

🧠 Explainable AI with SHAP Visuals

☁️ Cloud Deployment (AWS / Streamlit Cloud)

🏛️ RBI & SEBI Guideline RAG Corpus


⭐ Acknowledgements
LangChain — for RAG pipelines

Hugging Face — for open-source transformers

Streamlit — for rapid UI development

Yahoo Finance & Google News — for data APIs

University of Calcutta — for academic guidance

💬 “Finance AI Suite transforms data into intelligence, and intelligence into insight.”
— Sohan Ghosh (2025)
