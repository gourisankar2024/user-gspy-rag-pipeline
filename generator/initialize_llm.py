import os
from langchain_groq import ChatGroq

def initialize_llm():
    os.environ["GROQ_API_KEY"] = "your_groq_api_key"
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)
    return llm