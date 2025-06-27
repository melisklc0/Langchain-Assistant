from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List, Optional
import os


def get_relevant_documents(query_text: str, db, k: int = 3, threshold: float = 0.7):
    """Sorgu metni ile veritabanından ilgili belgeleri alır."""
    try:
        if not query_text:
            raise ValueError("Sorgu metni boş olamaz.")
        
        if k <= 0:
            raise ValueError("K değeri 0'dan büyük olmalıdır.")
        
        if threshold < 0 or threshold > 1:
            raise ValueError("Eşik değeri 0 ile 1 arasında olmalıdır.")
    except ValueError as e:
        print(f"Hata: {e}")
        return " "
        
    results = db.similarity_search_with_score(query_text, k=3)
    if len(results) == 0:
        print("Alakalı belgeler bulunamadı.")
        return " "

    try:
        context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    except Exception:
        # Eğer sonuçlar sadece doc ise:
        context = "\n\n---\n\n".join([doc.page_content for doc in results])

    return context

