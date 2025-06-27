from dotenv import load_dotenv
from modules.models import get_embedding_model, get_llm
from modules.doc_loader import load_pdfs
from modules.text_splitter import split_text
from modules.chroma import get_chroma_db
from modules.retriever import get_relevant_documents
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.output_parser import StrOutputParser


def main():
    data_path = r"D:\Üniversite\Internship-Studies\Langchain-Studies\Langchain-Assistant\docs\bilgisayar-aglari"
    chroma_path = r"D:\Üniversite\Internship-Studies\Langchain-Studies\Langchain-Assistant\vectorstore\chroma_db_with_metadata"
    
    llm = get_llm()
    documents = load_pdfs(data_path)
    chunks = split_text(documents)
    db = get_chroma_db(chroma_path, chunks)
    query = "Bilgisayar ağları nedir?"
    context_text = get_relevant_documents(query, db, 3, 0.5)


    prompt = ChatPromptTemplate.from_template(
        "Aşağıdaki bağlamı kullanarak soruyu yanıtla. Yalnızca bağlamda bulunan bilgileri kullan."
        " Eğer bilmiyorsan, 'Bilmiyorum' olarak cevap ver."
        " \n\n{context}\n\nSoru: {question}\n\nCevap:"
    )
    print("\nPrompt template created:")

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_text, "question": query})
    print("\nModel response:")
    print(response)


    


    
    


      

    

if __name__ == "__main__":
    load_dotenv()
    main()
