import os
import shutil
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from modules.doc_loader import load_pdfs
from modules.text_splitter import split_text
from modules.models import get_embedding_model


def get_chroma_db(chroma_path: str, data_path: str):
    """Chroma vectorstore'u döndürür."""
    embedding_function = get_embedding_model()

    while True:
        if os.path.exists(chroma_path):
            print(f"Var olan Chroma vectorstore bulundu, ne yapmak istersiniz?")
            print("[y] Eski veritabanını yükle")
            print("[s] Veritabanını sil ve yeniden oluştur")

            user_input = input("Seçiminiz (y/s): ").strip().lower()

            if user_input == 'y':
                print(f"Chroma vectorstore yükleniyor...")
                db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
                print(f"Chroma vectorstore başarıyla yüklendi.")
                return db
            elif user_input == 's':
                print(f"Chroma vectorstore temizleniyor...")
                clear_chroma_db(chroma_path)
                break
            else:
                print(f"Geçersiz giriş, lütfen 'y' veya 's' girin.")
                continue
        else:
            break       
    
    print(f"Chroma vectorstore oluşturuluyor.")
    documents = load_pdfs(data_path)
    chunks = split_text(documents)
    db = create_chroma_db(chroma_path, chunks)
    return db


def create_chroma_db(chroma_path: str, chunks: list[Document]):
    """Chroma vectorstore oluşturur."""
    embedding_function = get_embedding_model()

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
