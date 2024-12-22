import logging
import os
from langchain_groq import ChatGroq

def initialize_llm():
    os.environ["GROQ_API_KEY"] = "your_groq_api_key"
    model_name = "llama3-8b-8192"
    llm = ChatGroq(model=model_name, temperature=0.7)
    logging.info(f'Generation LLM {model_name} initialized')
    return llm

def initialize_validation_llm():
    os.environ["GROQ_API_KEY"] = "your_groq_api_key"
    model_name = "llama-3.1-8b-instant"
    llm = ChatGroq(model=model_name, temperature=0.7)
    logging.info(f'Validation LLM {model_name} initialized')
    return llm