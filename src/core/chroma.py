import os
import shutil
from typing import Callable
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.doc_loader import load_pdfs
from core.text_splitter import split_text
from core.models import get_embedding_model


def get_chroma_db(chroma_path: str):
    """Chroma vectorstore'u döndürür."""
    embedding_function = get_embedding_model()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    print(f"Chroma vectorstore başarıyla yüklendi.")
    return db


def create_chroma_db(chroma_path: str, data_path: str, loader_function: Callable[[str], list]):
    """Chroma vectorstore oluşturur."""
    embedding_function = get_embedding_model()

    while True:
        documents = loader_function(data_path)
        chunks = split_text(documents)

        if chunks:
            break
        print("Eklenecek yeni belge bulunamadı.")
        data_path = input("Yeni bir veri yolu girin: ")

    try:
        print(f"Chroma vectorstore oluşturuluyor.")
        db = Chroma.from_documents(
            chunks, embedding_function, persist_directory=chroma_path
        )
        print(f"Chroma vectorstore başarıyla kaydedildi.")
        return db
    except Exception as e:
        print(f"Chroma vectorstore oluşturulurken hata oluştu: {e}")
        return None


def clear_chroma_db(chroma_path: str):
    """Chroma vectorstore'u temizler."""
    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path)
            print(f"Chroma vectorstore başarıyla temizlendi.")
        except Exception as e:
            print(f"Chroma vectorstore temizlenirken hata oluştu: {e}")
    else:
        print(f"Chroma vectorstore bulunamadı, temizleme işlemi gereksiz.")


def add_data_to_chroma_db(db: Chroma, data_path: str, loader_function: Callable[[str], list]):
    """Var olan bir Chroma vectorstore'a yeni belgeler ekler."""

    documents = loader_function(data_path)
    chunks = split_text(documents)

    if not chunks:
        print("Eklenecek yeni belge bulunamadı.")
        return
    try:
        db.add_documents(chunks)
        print("Yeni belgeler başarıyla Chroma vectorstore'a eklendi.")
    except Exception as e:
        print(f"Belgeler eklenirken hata oluştu: {e}")

