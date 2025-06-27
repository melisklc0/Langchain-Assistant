from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from typing import List


"""
def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200):
    #Belgeleri chunk'lara böler
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)"""


def split_text(documents: List[Document]):
    """Belge metinlerini parçalara böler."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"{len(documents)} adet sayfa, {len(chunks)} adet parçaya bölündü.")
    return chunks

""" kullanım örneği
from modules.text_splitter import split_text    
from modules.doc_loader import load_document       

documents = load_documents()
chunks = split_text(documents)
"""