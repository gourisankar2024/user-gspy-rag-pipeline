
from generator.create_prompt import create_prompt
from generator.initialize_llm import initialize_llm

# Initialize the LLM
llm = initialize_llm()


# Function to extract attributes
def extract_attributes(question, relevant_docs, response):
    # Format documents into a string by accessing the `page_content` attribute of each Document
    formatted_documents = split_relevant_documents(relevant_docs)
    formatted_response = split_response(response)
    # print(f'Formatted documents: {formatted_documents}')
    attribute_prompt = create_prompt(formatted_documents, question, formatted_response)

    # Instead of using BaseMessage, pass the formatted prompt directly to invoke
    result = llm.invoke(attribute_prompt)

    return result


def split_relevant_documents(relevant_docs: list) -> list:
    split_relevant_documents_list = []
    for relevant_doc_index, relevant_doc in enumerate(relevant_docs):
        sentences = []
        for sentence_index, sentence in enumerate(relevant_doc.page_content.split(".")):
            sentences.append([str(relevant_doc_index)+chr(97 + sentence_index), sentence])
        split_relevant_documents_list.append(sentences)
    return split_relevant_documents_list


def split_response(response: str):
    sentences = []
    for sentence_index, sentence in enumerate(response.split(".")):
        sentences.append([chr(97 + sentence_index), sentence])
    return sentences
