import firebase_admin
from firebase_admin import credentials, firestore
from langchain_community.chat_message_histories import FirestoreChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r"D:\Üniversite\Langchain-Assistant\.env")

cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
print("Credential path:", cred_path)
print("Dosya var mı:", os.path.exists(cred_path))

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

client = firestore.client()

try:
    doc_ref = client.collection("test").document("doc1")
    doc_ref.set({"test": "başarılı"})
    print("Veri yazıldı.")
except Exception as e:
    print("Hata:", e)



PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
DEFAULT_COLLECTION = "chat_history"

def get_chat_history(session_id: str, collection: str = DEFAULT_COLLECTION, user_id: str = "default_user") -> FirestoreChatMessageHistory:
    """ Belirtilen session_id ile bir FirestoreChatMessageHistory nesnesi döner."""
    return FirestoreChatMessageHistory(
        session_id=session_id,
        user_id=user_id,
        collection_name=collection
    )


