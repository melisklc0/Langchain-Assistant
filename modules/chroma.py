import os
import shutil
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from modules.doc_loader import load_pdfs
from modules.text_splitter import split_text
from modules.models import get_embedding_model


def get_chroma_db(chroma_path: str, chunks: list[Document]):
    """Chroma vectorstore'u yükler veya oluşturur."""
    embedding_function = get_embedding_model()

    try:
        if os.path.exists(chroma_path):
            print(f"Var olan Chroma vectorstore bulundu, yükleniyor.")
            db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
            print(f"Chroma vectorstore başarıyla yüklendi.")
            return db
    except Exception as e:
        print(f"Chroma vectorstore yüklenirken hata oluştu: {e}")
        return None

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


""" kullanım örneği
from modules.doc_loader import load_pdfs
from modules.text_splitter import split_documents
from modules.chroma import get_chroma_db

data_path = r"D:\Üniversite\Internship-Studies\Langchain-Studies\Langchain-Assistant\docs\bilgisayar-aglari"
chroma_path = r"D:\Üniversite\Internship-Studies\Langchain-Studies\Langchain-Assistant\vectorstore\chroma_db_with_metadata"
documents = load_pdfs(data_path)
chunks = split_documents(documents)
db = get_chroma_db(chroma_path, chunks)
"""