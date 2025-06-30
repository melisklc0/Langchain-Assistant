import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

def get_llm():
    """LLM modelini döndürecek."""
    OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY ortam değişkeni ayarlanmamış.")
    try:
        llm = ChatOpenAI(
            model="mistralai/mistral-small-3.2-24b-instruct:free",
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_API_BASE,
            temperature=0.7
        )
    except Exception as e:
        raise RuntimeError(f"ChatOpenAI başlatılamadı: {e}")
    return llm


def get_embedding_model():
    """Embedding modelini döndürecek."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:
        raise RuntimeError(f"HuggingFaceEmbeddings başlatılamadı: {e}")
    return embedding_model
    
