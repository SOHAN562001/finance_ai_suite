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
| **Fraud Detection Assistant** | Detects transaction anomalies using unsupervised ML and computes severity/risk scores. | IsolationForest, scikit-learn, Streamlit |
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

#### Screenshots
![Chatbot Screenshot 1](https://github.com/SOHAN562001/finance_ai_suite/blob/main/assets/1.jpg)
![Chatbot Screenshot 2](https://github.com/SOHAN562001/finance_ai_suite/blob/main/assets/2.jpg)

---

### 2. Economic Report Generator
- Fetches live and historical data for indices such as **NIFTY 50**, **S&P 500**, **NASDAQ**, etc.  
- Computes returns, volatility, and correlation between multiple markets.  
- Visualizes performance trends using interactive charts.  
- Summarizes findings with **LLM-based narrative insights**.

_Example Insight:_  
Between Aug–Nov 2025, NIFTY 50 rose 4.1 %, correlating 0.77 with S&P 500 — indicating strong global co-movement.

#### Screenshots
![Economic Report 1](https://github.com/SOHAN562001/finance_ai_suite/blob/main/assets/Economic%20Report%20Generator/1.jpg)
![Economic Report 2](https://github.com/SOHAN562001/finance_ai_suite/blob/main/assets/Economic%20Report%20Generator/2.jpg)

---

### 3. Fraud Detection Assistant
- Accepts transaction datasets (Amount, Merchant, Timestamp, Location, etc.).  
- Detects anomalies via **IsolationForest** (unsupervised).  
- Calculates **Fraud Severity (0–100)** per transaction and aggregates a **Customer Risk Index**.  
- Provides visual analytics for merchant, time, and correlation patterns.

_Example Result:_  
5.6 % of transactions flagged as potentially fraudulent due to high-value spikes or unusual activity time.

#### Screenshots
![Fraud Detection 1](https://github.com/SOHAN562001/finance_ai_suite/blob/main/assets/Fraud%20Detection%20Assistant/1.1.jpg)
![Fraud Detection 2](https://github.com/SOHAN562001/finance_ai_suite/blob/main/assets/Fraud%20Detection%20Assistant/3jpg)
![Fraud Detection 3](https://github.com/SOHAN562001/finance_ai_suite/blob/main/assets/Fraud%20Detection%20Assistant/4.jpg)
![Fraud Detection 4](https://github.com/SOHAN562001/finance_ai_suite/blob/main/assets/Fraud%20Detection%20Assistant/5.jpg)
![Fraud Detection 5](https://github.com/SOHAN562001/finance_ai_suite/blob/main/assets/Fraud%20Detection%20Assistant/6.jpg)

---

### 4. News Summarizer & Media Bias Analyzer
- Collects financial news using **Yahoo Finance API** and **Google News RSS**.  
- Cleans and tokenizes text with **BeautifulSoup**.  
- Performs tone and polarity classification using **VADER Sentiment Analysis**.  
- Visualizes sentiment ratios and bias per media outlet.  
- Generates concise comparative summaries using a summarization model.

_Example Insight:_  
Reliance Industries received 61 % positive coverage, while Tata Motors had 44 % neutral tone — reflecting cautious investor sentiment.

#### Screenshots
![News Summarizer 1](https://github.com/SOHAN562001/finance_ai_suite/blob/main/assets/News%20Summarizer/1.jpg)
![News Summarizer 2](https://github.com/SOHAN562001/finance_ai_suite/blob/main/assets/News%20Summarizer/2.jpg)
![News Summarizer 3](https://github.com/SOHAN562001/finance_ai_suite/blob/main/assets/News%20Summarizer/3.jpg)

---

## Technology Stack

| Layer | Tools and Libraries |
|-------|---------------------|
| **Frontend UI** | Streamlit with custom CSS (dark theme, modular layout) |
| **Backend AI / NLP** | LangChain, Hugging Face Transformers |
| **Vector Database** | ChromaDB |
| **Data APIs** | Yahoo Finance (`yfinance`), Google News RSS |
| **Machine Learning / Analytics** | scikit-learn, Pandas, NumPy |
| **Visualization** | Matplotlib, Plotly |
| **Sentiment Analysis** | VADER (NLTK) |
| **Deployment** | Streamlit Cloud / AWS EC2 |

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
├── app.py
├── requirements.txt
│
├── modules/
│ ├── faq_chatbot.py
│ ├── economic_report.py
│ ├── fraud_detection.py
│ ├── news_summarizer.py
│ └── utils.py
│
├── data/
│ ├── bank_faqs.csv
│ ├── chroma_bank_faqs/
│ └── fraud_transactions_realistic.csv
│
└── assets/
├── Finance FAQ Chatbot/
├── Economic Report Generator/
├── Fraud Detection Assistant/
└── News Summarizer/

yaml
Copy code

---

## Installation and Setup

**1. Clone the Repository**
```bash
git clone https://github.com/SOHAN562001/finance_ai_suite.git
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

Evaluation Summary
Module	Status	Key Metric	Result
FAQ Chatbot	Completed	90 % Contextual Precision	Excellent
Economic Report	Completed	r > 0.75 Correlation	Excellent
News Summarizer	Completed	~85 % Sentiment Accuracy	Reliable
Fraud Detection	Completed	IsolationForest (Unsupervised)	Robust

Challenges and Solutions
Challenge	Solution
Yahoo API Rate Limits	Added fallback to Google News RSS
Chroma Cache Conflicts	Auto-cleared cache with unique session IDs
Visualization Lag	Optimized using st.cache_data()
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


“Merging data science, AI, and finance to build intelligent decision systems.”

