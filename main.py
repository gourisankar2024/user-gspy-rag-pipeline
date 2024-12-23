import logging
from data.load_dataset import load_data
from generator.compute_rmse_auc_roc_metrics import compute_rmse_auc_roc_metrics
from retriever.chunk_documents import chunk_documents
from retriever.embed_documents import embed_documents
from generator.generate_metrics import generate_metrics
from generator.initialize_llm import initialize_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting the RAG pipeline")
    data_set_name = 'covidqa'

    # Load the dataset
    dataset = load_data(data_set_name)
    logging.info("Dataset loaded")

    # Chunk the dataset
    chunk_size = 1000  # default value
    if data_set_name == 'cuad':
        chunk_size = 3000
    documents = chunk_documents(dataset, chunk_size)
    logging.info("Documents chunked")

    # Embed the documents
    vector_store = embed_documents(documents)
    logging.info("Documents embedded")
    
     # Initialize the Generation LLM
    llm = initialize_llm()

    # Sample question
    row_num = 10
    sample_question = dataset[row_num]['question']

    # Call generate_metrics for above sample question
    generate_metrics(llm, vector_store, sample_question)
    
    #Compute RMSE and AUC-ROC for entire dataset
    compute_rmse_auc_roc_metrics(llm, dataset, vector_store, 10)
    
    logging.info("Finished!!!")

if __name__ == "__main__":
    main()