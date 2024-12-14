from datasets import load_dataset

def load_data():
    return load_dataset("rungalileo/ragbench", 'covidqa', split="train")