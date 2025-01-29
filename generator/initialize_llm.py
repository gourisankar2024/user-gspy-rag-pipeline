import logging
import os
from langchain_groq import ChatGroq

def initialize_generation_llm():
    os.environ["GROQ_API_KEY"] = ""
    model_name = "llama3-8b-8192"
    llm = ChatGroq(model=model_name, temperature=0.7)
    logging.info(f'Generation LLM {model_name} initialized')
    return llm

def initialize_validation_llm():
    os.environ["GROQ_API_KEY"] = ""
    model_name = "llama3-70b-8192"
    llm = ChatGroq(model=model_name, temperature=0.7)
    logging.info(f'Validation LLM {model_name} initialized')
    return llm