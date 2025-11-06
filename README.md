# Finance AI Suite  
### An Integrated LLM + RAG–Powered Financial Analytics Platform  
**Author:** Sohan Ghosh  
_M.Sc. in Data Science & Artificial Intelligence, University of Calcutta (2025)_

---

## Overview

The **Finance AI Suite** is a unified, AI-driven financial analytics system integrating **Large Language Models (LLMs)**, **Retrieval-Augmented Generation (RAG)**, and **Machine Learning**.  
It delivers intelligent, explainable insights for finance and banking through a modern, modular **Streamlit** interface.

The project demonstrates how frameworks such as **LangChain**, **ChromaDB**, and **Hugging Face Transformers** can be combined into a production-ready, interactive platform for real-world financial intelligence.

---

## Modules Overview

| Module | Description | Core Technologies |
|--------|--------------|-------------------|
| **Finance FAQ Chatbot** | Answers 1,700+ banking and finance FAQs using vector similarity search and LLM reasoning. | LangChain, ChromaDB, Sentence Transformers, FLAN-T5 |
| **Economic Report Generator** | Generates analytical market reports and LLM-interpreted insights from global index data. | yFinance, Pandas, Plotly, LangChain |
| **Fraud Detection Assistant** | Detects transaction anomalies using unsupervised ML and computes severity/risk scores. | IsolationForest, Scikit-learn, Streamlit |
| **News Summarizer & Bias Analyzer** | Performs financial news sentiment and bias analysis with AI-generated summaries. | BeautifulSoup, VADER, Requests |

---

## Key Highlights

### 1. Finance FAQ Chatbot (RAG-based)
- Embeds 1,764 finance FAQs using **`all-MiniLM-L6-v2`**.  
- Stores embeddings in **ChromaDB** for semantic retrieval.  
- Generates context-aware answers through **FLAN-T5** via LangChain pipelines.  
- Replaces static rule-based FAQ systems with adaptive reasoning.

_Example:_  
**Q:** What is NEFT?  
**A:** NEFT is an electronic fund transfer system that operates on a deferred net settlement basis between bank branches in India.

---

### 2. Economic Report Generator
- Fetches live and historical data for indices such as **NIFTY 50**, **S&P 500**, **NASDAQ**, etc.  
- Computes returns, volatility, and correlation between multiple markets.  
- Visualizes performance trends using interactive charts.  
- Summarizes findings with **LLM-based narrative insights**.

_Example Insight:_  
Between Aug–Nov 2025, NIFTY 50 rose 4.1%, correlating 0.77 with S&P 500 — indicating strong global co-movement.

---

### 3. Fraud Detection Assistant
- Accepts transaction datasets (Amount, Merchant, Timestamp, Location, etc.).  
- Detects anomalies via **IsolationForest** (unsupervised).  
- Calculates **Fraud Severity (0–100)** per transaction and aggregates a **Customer Risk Index**.  
- Provides visual analytics for merchant, time, and correlation patterns.

_Example Result:_  
5.6% of transactions flagged as potentially fraudulent due to high-value spikes or unusual activity time.

---

### 4. News Summarizer & Media Bias Analyzer
- Collects financial news using **Yahoo Finance API** and **Google News RSS**.  
- Cleans and tokenizes text with **BeautifulSoup**.  
- Performs tone and polarity classification using **VADER Sentiment Analysis**.  
- Visualizes sentiment ratios and bias per media outlet.  
- Generates concise comparative summaries using a summarization model.

_Example Insight:_  
Reliance Industries received 61% positive coverage, while Tata Motors had 44% neutral tone — reflecting cautious investor sentiment.

---

## Technology Stack

| Layer | Tools and Libraries |
|-------|---------------------|
| **Frontend UI** | Streamlit with custom CSS (dark theme, modular layout) |
| **Backend AI/NLP** | LangChain, Hugging Face Transformers |
| **Vector Database** | ChromaDB |
| **Data APIs** | Yahoo Finance (`yfinance`), Google News RSS |
| **Machine Learning / Analytics** | Scikit-learn, Pandas, NumPy |
| **Visualization** | Matplotlib, Plotly |
| **Sentiment Analysis** | VADER (NLTK) |
| **Deployment Ready** | Streamlit Cloud / AWS EC2 |

---

## System Architecture

User Interface (Streamlit)
│
├── [Module 1] Finance FAQ Chatbot
│ └── LangChain + ChromaDB + LLM
│
├── [Module 2] Economic Report Generator
│ └── yFinance + LLM Summary
│
├── [Module 3] Fraud Detection Assistant
│ └── IsolationForest + Risk Scoring
│
└── [Module 4] News Summarizer
└── Yahoo/Google News + VADER Sentiment

yaml
Copy code

---

## Folder Structure

finance_ai_suite/
│
├── app.py # Main Streamlit entry point
├── requirements.txt # Dependencies
│
├── modules/
│ ├── faq_chatbot.py # RAG-based chatbot
│ ├── economic_report.py # Market analysis
│ ├── fraud_detection.py # Anomaly detection logic
│ ├── news_summarizer.py # Sentiment & bias analysis
│ └── utils.py # Helper utilities
│
├── data/
│ ├── bank_faqs.csv
│ ├── chroma_bank_faqs/
│ └── fraud_transactions_realistic.csv
│
└── README.md

yaml
Copy code

---

## Installation and Setup

**1. Clone the Repository**
```bash
git clone https://github.com/yourusername/finance_ai_suite.git
cd finance_ai_suite
2. Install Dependencies

bash
Copy code
pip install -r requirements.txt
3. Run the Streamlit App

bash
Copy code
streamlit run app.py
4. Open in Browser
http://localhost:8501

Screenshots (to be added)
Add 10–12 screenshots in this section after running the app:

Home Dashboard

Finance Chatbot Query

Economic Report Graphs

Fraud Detection Visualization

Customer Risk Index

News Sentiment Charts

Correlation Heatmap

AI Summary Block

Data Upload Interface

Overall System Interface

Evaluation Summary
Module	Status	Key Metric	Result
FAQ Chatbot	Completed	90% Contextual Precision	Excellent
Economic Report	Completed	r > 0.75 Correlation	Excellent
News Summarizer	Completed	~85% Sentiment Accuracy	Reliable
Fraud Detection	Completed	IsolationForest (unsupervised)	Robust

Challenges and Solutions
Challenge	Solution
Yahoo API Rate Limits	Added fallback to Google News RSS
Chroma Cache Conflicts	Auto-cleared cache and unique session IDs
Visualization Lag	Optimized with st.cache_data()
Model Memory Usage	Switched to lightweight FLAN-T5 models

Future Enhancements
Voice-enabled chatbot (Whisper + TTS integration)

Portfolio risk and asset analysis (Sharpe ratio, VaR)

Explainable AI dashboards (SHAP visualization)

Deployment on AWS EC2 or Streamlit Cloud

Domain-tuned RAG datasets (RBI, SEBI guidelines)

Author
Sohan Ghosh
M.Sc. in Data Science & Artificial Intelligence
University of Calcutta
📍 Kolkata, India
LinkedIn

“Merging data science, AI, and finance to build intelligent decision systems.”

License
This project is licensed under the MIT License.
You may reuse, modify, or extend it with proper attribution.

Acknowledgements
LangChain — for modular RAG pipelines

Hugging Face — for open-source transformer models

Streamlit — for rapid and interactive UI

Yahoo Finance & Google News — for open data APIs

University of Calcutta — for academic support and mentorship

“Finance AI Suite transforms data into intelligence, and intelligence into insight.”
— Sohan Ghosh (2025)
