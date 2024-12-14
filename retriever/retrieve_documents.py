def retrieve_top_k_documents(vector_store, query, top_k=5):
    return vector_store.similarity_search(query, k=top_k)