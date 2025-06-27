from pathlib import Path
from langchain.document_loaders import TextLoader, PyMuPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader


"""
def load_documents(file_path: str):
    ## Dosyaları yükleyecek
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Dosya yolu bulunamadı: {file_path}")

    files = list(Path(file_path).glob("*"))
    for file in files:
        dosya_uzantisi = file.suffix.lower()
        ## Dosya uzantısını al ve küçük harfe çevir
    
        if dosya_uzantisi == '.txt':
            loader = TextLoader(str(file), encoding='utf-8')
        elif dosya_uzantisi in ['.pdf', '.docx']:
            loader = PyMuPDFLoader(str(file))
        else:
            raise ValueError(f"Dosya tipi desteklenmiyor: {dosya_uzantisi}")
    
    return loader.load()"""


def load_pdfs(data_path: str):
    """ PDF dosyalarını yükleyecek fonksiyon """

    data_path = Path(data_path)
    # mesela burda path null dönerse ne olur
    # önlem // en saçma yerden gelebilecek hataları dene
    if not data_path.exists():
        raise FileNotFoundError(f"Dosya yolu bulunamadı: {data_path}")
    folder = data_path.name
    try:
        loader = PyPDFDirectoryLoader(data_path)
        # burda load yapamazsa ne olur
        documents = loader.load()
        # null bir şeyin length'ini alırsan ne hata verir
        print(f"{folder} klasörü başarıyla yüklendi, {len(documents)} adet sayfa bulundu.")
    except Exception as e:
        print(f"{folder} klasörü yüklenemedi: {e}")
    return documents
    # global exception handler 


""" kullanım örneği
from modules.doc_loader import load_pdfs

data_path = r"D:\Üniversite\Internship-Studies\Langchain-Studies\Langchain-Assistant\docs\bilgisayar-aglari"
load_pdfs(data_path)
""" 