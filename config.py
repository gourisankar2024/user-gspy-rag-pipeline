
class ConfigConstants:
    # Constants related to datasets and models
    DATA_SET_NAMES = ['covidqa', 'techqa', 'cuad']
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    RE_RANKER_MODEL_NAME = 'cross-encoder/ms-marco-electra-base'
    DEFAULT_CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

class AppConfig:
    def __init__(self, vector_store, gen_llm, val_llm):
        self.vector_store = vector_store
        self.gen_llm = gen_llm
        self.val_llm = val_llm