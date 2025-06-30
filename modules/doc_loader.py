from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader



def load_pdfs(data_path: str):
    """ PDF dosyalarını yükleyecek fonksiyon """

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dosya yolu bulunamadı: {data_path}")
    folder = data_path.name
    try:
        loader = PyPDFDirectoryLoader(data_path)
        documents = loader.load()
        print(f"{folder} klasörü başarıyla yüklendi, {len(documents)} adet sayfa bulundu.")
    except Exception as e:
        print(f"{folder} klasörü yüklenemedi: {e}")
    return documents



""" kullanım örneği
from modules.doc_loader import load_pdfs

data_path = r"D:\Üniversite\Internship-Studies\Langchain-Studies\Langchain-Assistant\docs\bilgisayar-aglari"
load_pdfs(data_path)
""" 