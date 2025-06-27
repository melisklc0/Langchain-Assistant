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

    if os.path.exists(chroma_path):
        print(f"Var olan Chroma vectorstore bulundu, siliniyor.")
        shutil.rmtree(chroma_path)
    
    print(f"Chroma vectorstore oluşturuluyor.")
    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=chroma_path
    )
    print(f"Chroma vectorstore başarıyla kaydedildi.")
    return db

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