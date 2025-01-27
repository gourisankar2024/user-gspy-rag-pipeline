import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import ConfigConstants

def embed_documents(documents, embedding_path="embeddings.faiss"):
    embedding_model = HuggingFaceEmbeddings(model_name=ConfigConstants.EMBEDDING_MODEL_NAME)
    
    if os.path.exists(embedding_path):
        logging.info("Loading embeddings from local file")
        vector_store = FAISS.load_local(embedding_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        logging.info("Generating and saving embeddings")
        vector_store = FAISS.from_texts([doc['text'] for doc in documents], embedding_model)
        vector_store.save_local(embedding_path)
    
    return vector_store
