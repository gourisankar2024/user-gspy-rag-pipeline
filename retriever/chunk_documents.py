from langchain.text_splitter import CharacterTextSplitter

def chunk_documents(dataset, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []
    for data in dataset:
        text_list = data['documents']
        for text in text_list:
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                documents.append({'text': chunk, 'source': f"{data['question']}_chunk_{i}"})
    return documents