import os
from dotenv import load_dotenv
load_dotenv(r"D:\Üniversite\Langchain-Assistant\.env")
USER_AGENT = os.getenv("USER_AGENT")

from modules.models import get_embedding_model, get_llm
from modules.doc_loader import load_pdfs, load_url, load_text_files
from modules.text_splitter import split_text
from modules.chroma import get_chroma_db, create_chroma_db, add_data_to_chroma_db
from modules.retriever import get_relevant_documents
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import chromadb


def main():

    data_path = r"D:\Üniversite\Internship-Studies\Langchain-Studies\Langchain-Assistant\docs\bilgisayar-aglari"
    chroma_path = r"D:\Üniversite\Internship-Studies\Langchain-Studies\Langchain-Assistant\vectorstore\chroma_db_with_metadata"

    llm = get_llm()

    if os.path.exists(chroma_path):
        db = get_chroma_db(chroma_path)
    else:
        db = create_chroma_db(chroma_path, data_path, load_pdfs)

    url_path = ["https://tr.wikipedia.org/wiki/Bilgisayar_a%C4%9F%C4%B1"]
    add_data_to_chroma_db(db, url_path, load_url)

    chat_history = [
        SystemMessage(
            content="Sen bir üniversitede asistansın." 
            "Öğrencilerden gelen sorulara az ve öz cevap veriyorsun."
            "İlgili bilgi sağlanan context'te bulunmuyorsa bilmiyorum de."
        )
    ]

    prompt = ChatPromptTemplate.from_template(
        "Sen bir üniversitede asistansın." 
        "Öğrencilerden gelen sorulara az ve öz cevap veriyorsun."
        "İlgili bilgi sağlanan context'te bulunmuyorsa bilmiyorum de."
        "\n\n{context}\n\nSoru: {question}\n\nCevap:"
    )

    chain = prompt | llm | StrOutputParser()

    while True:
        query = input("Soru (çıkmak için exit): ")
        if query.lower() == "exit":
            break

        context_text = get_relevant_documents(query, db)
        response = chain.invoke({"context": context_text, "question": query})

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=response))

        print(f"Cevap: {response} \n")

    print("\n---- Sohbet Geçmişi ----")
    for message in chat_history:
        print(f"{message.type.capitalize()}: {message.content}")

    



if __name__ == "__main__":
    main()
