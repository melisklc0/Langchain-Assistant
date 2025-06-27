from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

data_path = "docs/bilgisayar-aglari"

def load_documents():
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content) 
    return chunks

documents = load_documents()
print(f"Loaded {len(documents)} documents.")
chunks = split_text(documents) 
print(f"Split into {len(chunks)} chunks.")

def get_embedding_model():
    ## Embedding modelini döndürecek
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings
