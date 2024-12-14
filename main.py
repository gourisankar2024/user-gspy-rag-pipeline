import logging, json
from data.load_dataset import load_data
from retriever.chunk_documents import chunk_documents
from retriever.embed_documents import embed_documents
from retriever.retrieve_documents import retrieve_top_k_documents
from generator.initialize_llm import initialize_llm
from generator.generate_response import generate_response
from generator.extract_attributes import extract_attributes

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

    # Initialize the LLM
    llm = initialize_llm()
    logging.info("LLM initialized")

    # Sample question
    sample_question = dataset[0]['question']
    logging.info(f"Sample question: {sample_question}")

    # Retrieve relevant documents
    relevant_docs = retrieve_top_k_documents(vector_store, sample_question, top_k=5)
    logging.info("Relevant documents retrieved :", print(len(relevant_docs)))
    # Log each retrieved document individually
    #for i, doc in enumerate(relevant_docs):
        #logging.info(f"Relevant document {i+1}: {doc}")

    # Generate a response using the relevant documents
    response, source_docs = generate_response(llm, vector_store, sample_question)
    logging.info("Response generated")

    # Print the response
    print(f"Response: {response}")
    print(f"Source Documents: {source_docs}")

     # Extract attributes from the response and source documents
    attributes = extract_attributes(sample_question, relevant_docs, response)
    
    # Only proceed if the content is not empty
    if attributes.content:
        result_content = attributes.content  # Access the content attribute
        
        # Extract the JSON part from the result_content
        json_start = result_content.find("{")
        json_end = result_content.rfind("}") + 1
        json_str = result_content[json_start:json_end]
        
        try:
            result_json = json.loads(json_str)
            print(json.dumps(result_json, indent=2))
        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError: {e}")

if __name__ == "__main__":
    main()