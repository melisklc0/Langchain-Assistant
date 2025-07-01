from dotenv import load_dotenv
from modules.models import get_embedding_model, get_llm
from modules.doc_loader import load_pdfs
from modules.text_splitter import split_text
from modules.chroma import get_chroma_db
from modules.retriever import get_relevant_documents
from modules.firebase import get_chat_history
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os




def main():
    data_path = r"D:\Üniversite\Internship-Studies\Langchain-Studies\Langchain-Assistant\docs\bilgisayar-aglari"
    chroma_path = r"D:\Üniversite\Internship-Studies\Langchain-Studies\Langchain-Assistant\vectorstore\chroma_db_with_metadata"
    
    load_dotenv()
    llm = get_llm()
    db = get_chroma_db(chroma_path, data_path)

    session_id = input("Oturum adınızı giriniz: ").strip()
    chat_history = get_chat_history(session_id)

    if len(chat_history.messages) == 0:
        chat_history.add_message(SystemMessage(
            content="Sen bir üniversitede asistansın."
                    "Öğrencilerden gelen sorulara az ve öz cevap veriyorsun."
                    "Yalnızca sağlanan context'e göre cevap ver."
        ))


    prompt = ChatPromptTemplate.from_template(
        "Sen bir üniversitede asistansın." 
        "Öğrencilerden gelen sorulara az ve öz cevap veriyorsun."
        "Yalnızca sağlanan context'e göre cevap ver."
        "\n\n{context}\n\nSoru: {question}\n\nCevap:"
    )

    chain = prompt | llm | StrOutputParser()

    while True:
        query = input("Soru: ")
        if query.lower() == "çıkış":
            break

        context_text = get_relevant_documents(query, db, k=6)
        response = chain.invoke({"context": context_text, "question": query})
    
        chat_history.add_user_message(HumanMessage(content=query))
        chat_history.add_ai_message(AIMessage(content=response))

        print(f"Cevap: {response} \n")

    print("\n---- Firebase'den Sohbet Geçmişi ----")
    for message in chat_history.messages:
        print(f"{message.type.capitalize()}: {message.content}")


    
if __name__ == "__main__":
    main()
