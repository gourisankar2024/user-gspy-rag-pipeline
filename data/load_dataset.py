import logging
from datasets import load_dataset

def load_data(data_set_name):
    logging.info("Loading dataset")
    dataset = load_dataset("rungalileo/ragbench", data_set_name, split="test")
    logging.info("Dataset loaded successfully")
    logging.info(f"Number of documents found: {dataset.num_rows}")
    return dataset