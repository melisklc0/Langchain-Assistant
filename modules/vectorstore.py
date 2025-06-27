import os
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from pathlib import Path
from modules.text_splitter import split_documents

def get_vectorstore(chunk_size, chunk_overlap, embedding_model):
    # Klasör yolları
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(current_dir, "docs")
    db_dir = os.path.join(current_dir, "vectorstore")
    persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

    print(f"Belgeler klasörü: {docs_dir}")
    print(f"Vectorstore klasörü: {persistent_directory}")

    # Eğer vectorstore daha önce yoksa
    if os.path.exists(persistent_directory):
        print("Var olan vectorstore bulundu, yüklemeye hazır.")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_model())
        return db
    else:
        print("Vectorstore bulunamadı, yeni oluşturulacak.")

        # Belgeler klasörü yoksa hata ver
        if not os.path.exists(docs_dir):
            raise FileNotFoundError(f"{docs_dir} klasörü bulunamadı. PDF veya TXT dosyalarını buraya ekleyin.")

        # Dosyaları al
        files = list(Path(docs_dir).glob("*"))
        documents = []

        for file_path in files:
            if file_path.suffix == ".pdf":
                loader = PyMuPDFLoader(str(file_path))
            elif file_path.suffix == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                continue
            file_docs = loader.load()
            for doc in file_docs:
                doc.metadata = {"source": file_path.name}
                documents.append(doc)

        print(f"Eklenen toplam belge sayısı: {len(documents)}")

        # Chunk'la
        docs = split_documents(
            documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Chroma vectorstore oluştur
        print("Vectorstore oluşturuluyor...")
        db = Chroma.from_documents(docs, embedding_model(), persist_directory=persistent_directory)
        db.persist()
        print("Vectorstore başarıyla kaydedildi.")
        return db     

