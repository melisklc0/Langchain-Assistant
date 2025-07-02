import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import WebBaseLoader


def load_pdfs(data_path: str):
    """ PDF dosyalarını yükleyecek fonksiyon """

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dosya yolu bulunamadı: {data_path}")
    folder = data_path.name
    documents = []
    try:
        loader = PyPDFDirectoryLoader(data_path)
        documents = loader.load()
        print(f"{folder} klasörü başarıyla yüklendi, {len(documents)} adet sayfa bulundu.")
        return documents
    except Exception as e:
        print(f"{folder} klasörü yüklenemedi: {e}")
    return documents


def load_url(url_path: str):
    """ URL'den içerik yükleyecek fonksiyon """
    load_dotenv(r"D:\Üniversite\Langchain-Assistant\.env")
    user_agent = os.getenv("USER_AGENT")
    headers = {
        "User-Agent": user_agent
    }
    documents = []
    try:
        loader = WebBaseLoader(url_path, headers)
        documents = loader.load()
        print(f"{url_path} adresi başarıyla yüklendi, {len(documents)} adet sayfa bulundu.")
    except Exception as e:
        print(f"{url_path} adresi yüklenemedi: {e}")
    return documents


def load_text_files(data_path: str):
    """ Metin dosyalarını yükleyecek fonksiyon """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dosya yolu bulunamadı: {data_path}")
    folder = data_path.name
    documents = []
    try:
        loader = TextLoader(data_path, encoding="utf-8")
        documents = loader.load()
        print(f"{folder} klasörü başarıyla yüklendi, {len(documents)} adet sayfa bulundu.")
    except Exception as e:
        print(f"{folder} klasörü yüklenemedi: {e}")
    return documents
