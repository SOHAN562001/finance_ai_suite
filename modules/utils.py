# modules/utils.py
import os
from dotenv import load_dotenv

def load_api_keys():
    """
    Reads the .env file at project root and loads API keys into environment.
    Returns a tuple (hf_token, fmp_key, openai_key, google_key).
    """
    load_dotenv()  # This automatically finds and loads the .env file
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    fmp_key = os.getenv("FMP_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    return hf_token, fmp_key, openai_key, google_key
