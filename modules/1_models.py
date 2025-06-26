import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings

def get_llm():
    ## LLM modeli döndürecek
    OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
    llm = ChatOpenAI(
        model="mistralai/mistral-small-3.2-24b-instruct:free",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=OPENROUTER_API_BASE,
        temperature=0.7
    )
    
    return llm

def get_embedding_model():
    ## Embedding modelini döndürecek
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_model
    
