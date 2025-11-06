# modules/fraud_detection.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from transformers import pipeline


# ----------------------------- Utilities -----------------------------
def _normalize_0_100(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to 0..100 robustly."""
    a_min, a_max = np.min(arr), np.max(arr)
    if np.isclose(a_min, a_max):
        return np.zeros_like(arr)
    return (arr - a_min) / (a_max - a_min) * 100.0


# ------------------------- Main App Entrypoint ------------------------
def run_fraud_detection():
    st.title("🕵️‍♂️ Fraud Detection Assistant")
    st.caption("Upload → Detect anomalies → Visualize → Severity scoring → Customer risk → AI insights")

    st.markdown(
        """
**Workflow**
1) 📂 Upload CSV  
2) ⚙️ Isolation Forest anomaly detection  
3) 📊 Visual dashboards (scatter / merchants / correlation / timeline)  
4) 🎯 **Fraud Severity (0–100)** per transaction  
5) 👥 **Customer Risk Index** (aggregated)  
6) 🧠 AI summary
"""
    )

    # ---------- File Upload ----------
    uploaded_file = st.file_uploader("📂 Upload your transaction CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file to begin.")
        return

    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        return

    # ---------- Feature Selection ----------
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found for analysis.")
        return

    features = st.multiselect(
        "Select numeric columns for anomaly detection:",
        numeric_cols,
        default=numeric_cols[: min(5, len(numeric_cols))],
        help="Pick signal-heavy columns like Amount, Balance, Num_Transactions, Days_Since_Last"
    )
    if not features:
        st.warning("Select at least one numeric feature.")
        return

    # ---------- Model Parameters ----------
    contamination = st.slider("Expected anomaly ratio (fraud %):", 0.001, 0.2, 0.05)
    st.write(f"📊 Using contamination level: **{contamination*100:.1f}%**")

    # ---------- Detection ----------
    try:
        work = df.copy()
        X = work[features].fillna(work[features].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=42
        )
        preds = model.fit_predict(X_scaled)                 # -1 = anomaly, 1 = normal
        decision = model.decision_function(X_scaled)        # higher = more normal
        score_samples = model.score_samples(X_scaled)       # lower = more anomalous

        # Flags
        work["Fraud_Flag"] = (preds == -1).astype(int)

        # Severity (0..100): invert normality → normalize
        # Use negative score_samples (lower → more anomalous) to make higher = worse
        raw_anom = -score_samples
        work["Fraud_Severity"] = _normalize_0_100(raw_anom)

        # Nice summary
        fraud_cases = work[work["Fraud_Flag"] == 1]
        st.subheader("📈 Detection Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", len(work))
        c2.metric("Suspicious Cases", len(fraud_cases))
        c3.metric("Flagged %", f"{(len(fraud_cases) / len(work) * 100):.2f}%")

        st.subheader("🚨 Flagged Transactions (Top 10 by Severity)")
        if len(fraud_cases) > 0:
            st.dataframe(
                fraud_cases.sort_values("Fraud_Severity", ascending=False).head(10)
            )
        else:
            st.info("No suspicious rows flagged at current contamination level.")

    except Exception as e:
        st.error(f"Error during detection: {e}")
        return

    # ----------------------- Customer Risk Index -----------------------
    st.subheader("👥 Customer Risk Index")

    # Try to locate an ID column
    id_col_guess = None
    for cand in ["Customer_ID", "CustomerId", "customer_id", "ACCOUNT_ID", "Account_ID"]:
        if cand in work.columns:
            id_col_guess = cand
            break

    if id_col_guess is None:
        st.warning(
            "No customer identifier column found (e.g., 'Customer_ID'). "
            "Customer Risk Index will be skipped. "
            "Add a column named 'Customer_ID' to enable this feature."
        )
        customer_risk = None
    else:
        grp = work.groupby(id_col_guess, as_index=False).agg(
            Txn_Count=("Fraud_Flag", "size"),
            Fraud_Count=("Fraud_Flag", "sum"),
            Mean_Severity=("Fraud_Severity", "mean"),
            Max_Severity=("Fraud_Severity", "max"),
            Total_Amount=("Amount", "sum") if "Amount" in work.columns else ("Fraud_Flag", "sum"),
        )
        grp["Fraud_Rate_%"] = np.where(grp["Txn_Count"] > 0, grp["Fraud_Count"] / grp["Txn_Count"] * 100, 0.0)

        # Composite **Customer Risk Index** (0..100)
        # Blend mean severity & fraud rate; weight can be tuned
        grp["Customer_Risk_Index"] = 0.6 * grp["Mean_Severity"] + 0.4 * grp["Fraud_Rate_%"]
        # Normalize to 0..100 for a clean scale
        grp["Customer_Risk_Index"] = _normalize_0_100(grp["Customer_Risk_Index"].values)

        customer_risk = grp.sort_values("Customer_Risk_Index", ascending=False)
        st.dataframe(customer_risk.head(10))

        # Visuals
        st.markdown("**Top Risky Customers**")
        top_n = st.slider("Show top N customers:", 5, min(50, len(customer_risk)) if len(customer_risk) > 5 else 5, 10)
        show_cols = [id_col_guess, "Customer_Risk_Index", "Mean_Severity", "Fraud_Rate_%", "Txn_Count"]
        to_plot = customer_risk.head(top_n)[show_cols]

        plt.figure(figsize=(9, 4))
        plt.bar(to_plot[id_col_guess].astype(str), to_plot["Customer_Risk_Index"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Risk Index (0–100)")
        plt.title("Top Customer Risk Index")
        st.pyplot(plt.gcf())

        # Download button
        st.download_button(
            "💾 Download Customer Risk Table (CSV)",
            customer_risk.to_csv(index=False).encode("utf-8"),
            "customer_risk_index.csv",
            "text/csv",
        )

    # ----------------------- Visual Analytics -------------------------
    st.subheader("📊 Fraud Analytics Dashboard")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Amount vs Balance",
        "🏪 Merchant Trends",
        "📈 Correlation Map",
        "📆 Fraud Timeline",
        "🎯 Severity Distribution"
    ])

    # Tab 1: Scatter
    with tab1:
        if "Amount" in work.columns and "Balance" in work.columns:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(
                data=work,
                x="Amount",
                y="Balance",
                hue="Fraud_Flag",
                palette={0: "blue", 1: "red"},
                alpha=0.7
            )
            plt.title("Fraud vs Normal: Amount vs Balance")
            st.pyplot(plt.gcf())
        else:
            st.warning("Amount/Balance columns not found for scatter plot.")

    # Tab 2: Merchant Trends
    with tab2:
        if "Merchant_Category" in work.columns:
            plt.figure(figsize=(9, 4))
            sns.countplot(
                data=work,
                x="Merchant_Category",
                hue="Fraud_Flag",
                palette="Set2"
            )
            plt.xticks(rotation=45, ha="right")
            plt.title("Merchant Category vs Fraud Flag")
            st.pyplot(plt.gcf())
        else:
            st.warning("Merchant_Category column not found in dataset.")

    # Tab 3: Correlation
    with tab3:
        corr = work[features + ["Fraud_Flag", "Fraud_Severity"]].corr(numeric_only=True)
        plt.figure(figsize=(9, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap (incl. flags & severity)")
        st.pyplot(plt.gcf())

    # Tab 4: Timeline
    with tab4:
        if "Timestamp" in work.columns:
            try:
                work["Timestamp"] = pd.to_datetime(work["Timestamp"], errors="coerce")
                timeline = work.groupby(pd.Grouper(key="Timestamp", freq="D"))["Fraud_Flag"].sum().reset_index()
                plt.figure(figsize=(9, 4))
                sns.lineplot(
                    data=timeline,
                    x="Timestamp",
                    y="Fraud_Flag",
                    marker="o",
                    color="red"
                )
                plt.title("Fraud Cases Over Time")
                plt.xlabel("Date")
                plt.ylabel("Number of Fraudulent Transactions")
                plt.xticks(rotation=45)
                st.pyplot(plt.gcf())
            except Exception as e:
                st.warning(f"Timeline plot skipped: {e}")
        else:
            st.warning("Timestamp column not found in dataset.")

    # Tab 5: Severity Distribution
    with tab5:
        plt.figure(figsize=(9, 4))
        sns.histplot(work["Fraud_Severity"], bins=30, kde=True)
        plt.title("Fraud Severity Distribution (0–100)")
        plt.xlabel("Severity")
        plt.ylabel("Frequency")
        st.pyplot(plt.gcf())

        if id_col_guess:
            st.markdown("**Most Severe Recent Transactions**")
            cols_show = [id_col_guess, "Fraud_Flag", "Fraud_Severity"]
            extra = [c for c in ["Transaction_ID", "Amount", "Balance", "Merchant_Category", "Timestamp"] if c in work.columns]
            cols_show = extra + cols_show
            st.dataframe(
                work.sort_values("Fraud_Severity", ascending=False)[cols_show].head(10)
            )

    # -------------------------- AI Summary ----------------------------
    st.subheader("🧠 AI-Powered Fraud Insights")
    try:
        summarizer = pipeline("text2text-generation", model="google/flan-t5-small")
        top_merchants = (
            work.loc[work["Fraud_Flag"] == 1, "Merchant_Category"]
            .value_counts()
            .head(5)
            .to_dict()
            if "Merchant_Category" in work.columns
            else {}
        )
        summary_prompt = (
            f"Summarize fraud detection results:\n"
            f"- Total rows: {len(work)}, flagged: {work['Fraud_Flag'].sum()} ({work['Fraud_Flag'].mean()*100:.2f}%).\n"
            f"- Features: {', '.join(features)}.\n"
            f"- Average flagged severity: {work.loc[work['Fraud_Flag']==1, 'Fraud_Severity'].mean():.1f}.\n"
            f"- Top risky merchants (count): {top_merchants}.\n"
            f"- Explain key behaviors (amount spikes, long inactivity, high-risk merchants) in 4-6 bullet points."
        )
        ai_summary = summarizer(summary_prompt, max_new_tokens=220)[0]["generated_text"]
        st.success("✅ AI Summary Generated")
        st.write(ai_summary)
    except Exception as e:
        st.warning(f"⚠️ AI summarization skipped: {e}")
        ai_summary = "AI summary unavailable."

    # ----------------------- Downloads -------------------------------
    st.download_button(
        "💾 Download CSV with Flags & Severity",
        work.to_csv(index=False).encode("utf-8"),
        "fraud_detection_results.csv",
        "text/csv"
    )

    # Customer risk CSV (if available)
    if id_col_guess and isinstance(customer_risk, pd.DataFrame):
        st.download_button(
            "💾 Download Customer Risk Index (CSV)",
            customer_risk.to_csv(index=False).encode("utf-8"),
            "customer_risk_index.csv",
            "text/csv"
        )

    report_text = f"""
🕵️‍♂️ FRAUD DETECTION REPORT
=============================
Records Analyzed: {len(work)}
Flagged Suspicious: {int(work['Fraud_Flag'].sum())} ({work['Fraud_Flag'].mean()*100:.2f}%)
Features Used: {', '.join(features)}

Severity:
- Mean Severity (flagged): {work.loc[work['Fraud_Flag']==1, 'Fraud_Severity'].mean():.2f}
- Max Severity: {work['Fraud_Severity'].max():.2f}

Customer Risk Index:
- Available: {"Yes" if id_col_guess else "No (missing Customer_ID)"}

AI Summary:
{ai_summary}
"""
    st.download_button(
        label="📄 Download Summary Report (TXT)",
        data=report_text,
        file_name="fraud_detection_report.txt"
    )
