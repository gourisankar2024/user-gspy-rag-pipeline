import logging
from generator.generate_response import generate_response
from retriever.retrieve_documents import retrieve_top_k_documents
from generator.compute_metrics import get_metrics
from generator.extract_attributes import extract_attributes

def generate_metrics(llm, vector_store, query):
    logging.info(f'Query: {query}')
    
    # Step 1: Retrieve relevant documents for given query
    relevant_docs = retrieve_top_k_documents(vector_store, query, top_k=5)
    logging.info(f"Relevant documents retrieved :{len(relevant_docs)}")

    # Log each retrieved document individually
    #for i, doc in enumerate(relevant_docs):
        #logging.info(f"Relevant document {i+1}: {doc} \n")

    # Step 2: Generate a response using LLM
    response, source_docs = generate_response(llm, vector_store, query, relevant_docs)

    logging.info(f"Response from LLM: {response}")

    # Step 3: Extract attributes and total sentences for each query
    attributes, total_sentences = extract_attributes(query, source_docs, response)

    # Call the get_metrics
    metrics = get_metrics(attributes, total_sentences)

    return metrics