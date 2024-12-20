import logging
from data.load_dataset import load_data
from generator import compute_rmse_auc_roc_metrics
from retriever.chunk_documents import chunk_documents
from retriever.embed_documents import embed_documents
from retriever.retrieve_documents import retrieve_top_k_documents
from generator.initialize_llm import initialize_llm
from generator.generate_response import generate_response
from generator.extract_attributes import extract_attributes
from generator.compute_metrics import get_metrics 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting the RAG pipeline")

    # Load the dataset
    dataset = load_data()
    logging.info("Dataset loaded")

    # Chunk the dataset
    documents = chunk_documents(dataset)
    logging.info("Documents chunked")

    # Embed the documents
    vector_store = embed_documents(documents)
    logging.info("Documents embedded")
    
    # Sample question
    row_num = 1
    sample_question = dataset[row_num]['question']
    logging.info(f"Sample question: {sample_question}")

    # Retrieve relevant documents
    relevant_docs = retrieve_top_k_documents(vector_store, sample_question, top_k=5)
    logging.info(f"Relevant documents retrieved :{len(relevant_docs)}")
    # Log each retrieved document individually
    #for i, doc in enumerate(relevant_docs):
        #logging.info(f"Relevant document {i+1}: {doc} \n")

    # Initialize the LLM
    llm = initialize_llm()
    logging.info("LLM initialized")

    # Generate a response using the relevant documents
    response, source_docs = generate_response(llm, vector_store, sample_question, relevant_docs)
    logging.info("Response generated")

    # Print the response
    logging.info(f"Response from LLM: {response}")
    #print(f"Source Documents: {source_docs}")

    # Valuations : Extract attributes from the response and source documents
    attributes, total_sentences = extract_attributes(sample_question, source_docs, response)
    
    # Call the process_attributes method in the main block
    metrics = get_metrics(attributes, total_sentences)
    
    #Compute RMSE and AUC-ROC for entire dataset
    #compute_rmse_auc_roc_metrics(llm, dataset, vector_store)
   
if __name__ == "__main__":
    main()