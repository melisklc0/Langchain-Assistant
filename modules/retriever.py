from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List, Optional
import os

def get_relevant_documents(query_text: str, db, k: int = 3, threshold: float = 0.7):
    """Sorgu metni ile veritabanından ilgili belgeleri alır."""
    try:
        results = db.similarity_search_with_score(query_text, k=k)
    except Exception as e:
        print(f"Veritabanı sorgusu sırasında hata oluştu: {e}")
        return " "

    if not results or len(results) == 0:
        print("Alakalı belgeler bulunamadı.")
        return " "

    try:
        context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    except Exception as e:
        try:
            # Eğer sonuçlar sadece doc ise:
            context = "\n\n---\n\n".join([doc.page_content for doc in results])
        except Exception as inner_e:
            print(f"Belgeler işlenirken hata oluştu: {inner_e}")
            return " "

    return context

