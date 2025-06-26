from pathlib import Path
from langchain.document_loaders import TextLoader, PyMuPDFLoader

def load_documents(file_path: str):
    ## Dosyaları yükleyecek
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Dosya yolu bulunamadı: {file_path}")

    file_extension = Path(file_path).suffix.lower()
    ## Dosya uzantısını al ve küçük harfe çevir
    
    if file_extension == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    elif file_extension in ['.pdf', '.docx']:
        loader = PyMuPDFLoader(file_path)
    else:
        raise ValueError(f"Dosya tipi desteklenmiyor: {file_extension}")
    
    return loader.load()
