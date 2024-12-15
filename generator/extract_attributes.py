from generator.create_prompt import create_prompt
from generator.initialize_llm import initialize_llm
from generator.document_utils import Document, apply_sentence_keys_documents, apply_sentence_keys_response

# Initialize the LLM
llm = initialize_llm()

# Function to extract attributes
def extract_attributes(question, relevant_docs, response):
    # Format documents into a string by accessing the `page_content` attribute of each Document
    #formatted_documents = "\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(relevant_docs)])
    formatted_documents = apply_sentence_keys_documents(relevant_docs)
    formatted_responses = apply_sentence_keys_response(response)

    # Calculate the total number of sentences from formatted_documents
    total_sentences = sum(len(doc) for doc in formatted_documents)

    attribute_prompt = create_prompt(formatted_documents, question, formatted_responses)

    # Instead of using BaseMessage, pass the formatted prompt directly to invoke
    result = llm.invoke(attribute_prompt)

    return result, total_sentences