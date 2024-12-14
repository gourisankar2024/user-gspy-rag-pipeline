from langchain.chains import RetrievalQA

def generate_response(llm, vector_store, question):
    # Create a retrieval-based question-answering chain using the relevant documents
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    result = qa_chain.invoke(question)
    response = result['result']
    source_docs = result['source_documents']
    return response, source_docs