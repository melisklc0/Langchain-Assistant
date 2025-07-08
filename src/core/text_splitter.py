from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from typing import List

def split_text(documents: List[Document]):
    """Belge metinlerini parçalara böler."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"{len(documents)} adet sayfa, {len(chunks)} adet parçaya bölündü.")
        return chunks
    except Exception as e:
        print(f"Metin parçalama sırasında bir hata oluştu: {e}")
        return []

"""
Kullanım örneği:
from src.core.text_splitter import split_text
from src.core.doc_loader import load_documents

documents = load_documents()
chunks = split_text(documents)
"""
