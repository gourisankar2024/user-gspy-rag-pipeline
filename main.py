from data.load_dataset import load_data
from retriever.chunk_documents import chunk_documents
from retriever.embed_documents import embed_documents
from retriever.retrieve_documents import retrieve_top_k_documents
from generator.initialize_llm import initialize_llm
from generator.generate_response import generate_response

def main():
    # Load the dataset
    dataset = load_data()
    
    # Chunk the dataset
    documents = chunk_documents(dataset)
    
    # Embed the documents
    vector_store = embed_documents(documents)
    
    # Initialize the LLM
    llm = initialize_llm()
    
    # Sample question
    sample_question = dataset[0]['question']
    
    # Retrieve relevant documents
    relevant_docs = retrieve_top_k_documents(vector_store, sample_question, top_k=5)
    
    # Generate a response
    response, source_docs = generate_response(llm, vector_store, sample_question)
    
    # Print the response
    print(f"Response: {response}")
    print(f"Source Documents: {source_docs}")

if __name__ == "__main__":
    main()