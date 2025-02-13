
class ConfigConstants:
    # Constants related to datasets and models
    DATA_SET_NAMES = ['covidqa', 'cuad', 'techqa']#, 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa']
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    RE_RANKER_MODEL_NAME = 'cross-encoder/ms-marco-electra-base'
    GENERATION_MODEL_NAME = 'mixtral-8x7b-32768'
    VALIDATION_MODEL_NAME = 'llama3-70b-8192'
    DEFAULT_CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

class AppConfig:
    def __init__(self, vector_store, gen_llm, val_llm):
        self.vector_store = vector_store
        self.gen_llm = gen_llm
        self.val_llm = val_llm