import logging
import os
from langchain_groq import ChatGroq

from config import ConfigConstants

def initialize_generation_llm():
    os.environ["GROQ_API_KEY"] = ""
    model_name = ConfigConstants.GENERATION_MODEL_NAME
    llm = ChatGroq(model=model_name, temperature=0.7)
    llm.name = model_name
    logging.info(f'Generation LLM {model_name} initialized')
    return llm

def initialize_validation_llm():
    os.environ["GROQ_API_KEY"] = ""
    model_name = ConfigConstants.VALIDATION_MODEL_NAME
    llm = ChatGroq(model=model_name, temperature=0.7)
    llm.name = model_name
    logging.info(f'Validation LLM {model_name} initialized')
    return llm