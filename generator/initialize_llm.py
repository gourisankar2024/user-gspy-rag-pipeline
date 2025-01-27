import logging
import os
from langchain_groq import ChatGroq

def initialize_generation_llm():
    os.environ["GROQ_API_KEY"] = "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"
    model_name = "mixtral-8x7b-32768"
    llm = ChatGroq(model=model_name, temperature=0.7)
    llm.name = model_name
    logging.info(f'Generation LLM {model_name} initialized')
    return llm

def initialize_validation_llm():
    os.environ["GROQ_API_KEY"] = "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"
    model_name = "llama3-70b-8192"
    llm = ChatGroq(model=model_name, temperature=0.7)
    llm.name = model_name
    logging.info(f'Validation LLM {model_name} initialized')
    return llm