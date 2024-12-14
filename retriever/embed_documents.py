from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def embed_documents(documents):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts([doc['text'] for doc in documents], embedding_model)
    return vector_store
