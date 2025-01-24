import logging
from data.load_dataset import load_data
from generator.compute_rmse_auc_roc_metrics import compute_rmse_auc_roc_metrics
from retriever.chunk_documents import chunk_documents
from retriever.embed_documents import embed_documents
from generator.generate_metrics import generate_metrics
from generator.initialize_llm import initialize_generation_llm
from generator.initialize_llm import initialize_validation_llm

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
    gen_llm = initialize_generation_llm()

    # Initialize the Validation LLM
    val_llm = initialize_validation_llm()

    # Sample question
    row_num = 2
    query = dataset[row_num]['question']

    # Call generate_metrics for above sample question
    #generate_metrics(gen_llm, val_llm, vector_store, query)
    
    #Compute RMSE and AUC-ROC for entire dataset
    compute_rmse_auc_roc_metrics(gen_llm, val_llm, dataset, vector_store, 10)
    
    logging.info("Finished!!!")

if __name__ == "__main__":
    main()