import logging
from data.load_dataset import load_data
from generator.compute_rmse_auc_roc_metrics import compute_rmse_auc_roc_metrics
from retriever.chunk_documents import chunk_documents
from retriever.embed_documents import embed_documents
from generator.initialize_llm import initialize_generation_llm
from generator.initialize_llm import initialize_validation_llm
from app import launch_gradio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting the RAG pipeline")
    

    # Load single dataset
    #dataset = load_data(data_set_name)
    #logging.info("Dataset loaded")
    # List of datasets to load
    data_set_names = ['covidqa', 'techqa', 'cuad']

    default_chunk_size = 1000
    chunk_overlap = 200

    # Dictionary to store chunked documents
    all_chunked_documents = []
    # Load multiple datasets
    datasets = {}
    for data_set_name in data_set_names:
        logging.info(f"Loading dataset: {data_set_name}")
        datasets[data_set_name] = load_data(data_set_name)

        # Set chunk size based on dataset name
        chunk_size = default_chunk_size
        if data_set_name == 'cuad':
            chunk_size = 4000  # Custom chunk size for 'cuad'
        
        # Chunk documents
        chunked_documents = chunk_documents(datasets[data_set_name], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunked_documents.extend(chunked_documents)  # Combine all chunks

    # Access individual datasets
    #for name, dataset in datasets.items():
        #logging.info(f"Loaded {name} with {dataset.num_rows} rows")
    
    # Logging final count
    logging.info(f"Total chunked documents: {len(all_chunked_documents)}")
    
    # Embed the documents
    vector_store = embed_documents(all_chunked_documents)
    logging.info("Documents embedded")
    
     # Initialize the Generation LLM
    gen_llm = initialize_generation_llm()

    # Initialize the Validation LLM
    val_llm = initialize_validation_llm()

    #Compute RMSE and AUC-ROC for entire dataset
    data_set_name = 'covidqa'
    #compute_rmse_auc_roc_metrics(gen_llm, val_llm, datasets[data_set_name], vector_store, 10)
    
    # Launch the Gradio app
    launch_gradio(vector_store, gen_llm, val_llm)

    logging.info("Finished!!!")

if __name__ == "__main__":
    main()