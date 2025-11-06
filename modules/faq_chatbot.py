import os
import shutil
import streamlit as st
import pandas as pd
import torch

from modules.utils import load_api_keys

# ---- LangChain / HuggingFace imports ----
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


# ==================== CONFIG ====================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
try:
    torch.set_num_threads(2)
except Exception:
    pass


# ==================== HELPERS ====================
@st.cache_resource(show_spinner=False)
def _build_vectorstore_from_texts(texts: list[str]) -> Chroma:
    """Safely rebuild a Chroma vectorstore from texts, deleting corrupt DB if needed."""
    persist_dir = "data/chroma_bank_faqs"
    # If previous run left a corrupt index, nuke it cleanly
    if os.path.exists(persist_dir):
        try:
            _ = Chroma(persist_directory=persist_dir)
        except Exception:
            shutil.rmtree(persist_dir, ignore_errors=True)

    docs = [Document(page_content=t) for t in texts]
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="bank_faqs",
        persist_directory=persist_dir,
    )


@st.cache_resource(show_spinner=False)
def _load_llm_pipeline() -> HuggingFacePipeline:
    """Load a compact seq2seq LLM for CPU-only inference."""
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    gen_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        truncation=True,
        device=-1,  # CPU
    )
    return HuggingFacePipeline(pipeline=gen_pipe)


# ==================== MAIN CHATBOT ====================
def run_chatbot():
    st.header("🤖 Finance FAQ Chatbot")
    st.caption("Ask questions from Bank FAQs using Retrieval-Augmented Generation (RAG).")

    load_api_keys()

    data_path = "data/BankFAQs.csv"
    if not os.path.exists(data_path):
        st.error("❌ 'data/BankFAQs.csv' not found. Please place it under /data folder.")
        return

    df = pd.read_csv(data_path)
    df["content"] = df.apply(lambda r: f"Q: {r['Question']}\nA: {r['Answer']}", axis=1)
    st.success(f"✅ Loaded {len(df)} FAQs")

    # ---- Vectorstore ----
    with st.spinner("🔍 Building embeddings (first run may take ~30 s)..."):
        vectorstore = _build_vectorstore_from_texts(df["content"].tolist())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # ---- Prompt ----
    prompt = PromptTemplate(
        input_variables=["input_documents", "question"],
        template=(
            "You are a helpful banking assistant.\n"
            "Answer the question using only the information provided below.\n"
            "If the answer is not found, say: \"I'm not sure about that.\"\n\n"
            "Documents:\n{input_documents}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
    )

    # ---- Chain ----
    llm = _load_llm_pipeline()
    docs_chain = create_stuff_documents_chain(
        llm,
        prompt,
        document_variable_name="input_documents",
    )

    def answer_with_rag(question: str) -> str:
        docs = retriever.invoke(question)
        result = docs_chain.invoke({"input_documents": docs, "question": question})
        return result

    # ---- UI ----
    st.divider()
    user_q = st.text_input("Ask your banking question (e.g., What is NEFT?)")

    if user_q:
        with st.spinner("🤔 Thinking..."):
            try:
                answer = answer_with_rag(user_q)
                st.success("✅ Answer:")
                st.write(answer)

                with st.expander("📎 Retrieved context"):
                    docs = retriever.invoke(user_q)
                    for i, d in enumerate(docs, 1):
                        text = getattr(d, "page_content", str(d))
                        st.markdown(f"**Chunk {i}:**\n\n{text}")

            except OSError as e:
                st.error(
                    "⚠️ Low virtual-memory condition. "
                    "Close heavy apps or increase the Windows paging file size."
                )
                st.exception(e)
            except Exception as e:
                st.error("Something went wrong while generating the answer:")
                st.exception(e)
