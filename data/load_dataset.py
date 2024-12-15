import logging
from datasets import load_dataset

def load_data():
    logging.info("Loading dataset")
    dataset = load_dataset("rungalileo/ragbench", 'covidqa', split="test")
    logging.info("Dataset loaded successfully")
    logging.info(dataset)
    return dataset