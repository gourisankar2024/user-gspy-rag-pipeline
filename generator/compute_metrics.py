import json
import logging

def compute_metrics(attributes, total_sentences):
    # Extract relevant information from attributes
    all_relevant_sentence_keys = attributes.get("all_relevant_sentence_keys", [])
    all_utilized_sentence_keys = attributes.get("all_utilized_sentence_keys", [])
    sentence_support_information = attributes.get("sentence_support_information", [])

    # Compute Context Relevance
    context_relevance = len(all_relevant_sentence_keys) / total_sentences if total_sentences else 0
    
    # Compute Context Utilization
    context_utilization = len(all_utilized_sentence_keys) / total_sentences if total_sentences else 0
    
    # Compute Completeness score
    Ri = set(all_relevant_sentence_keys)
    Ui = set(all_utilized_sentence_keys)

    completeness_score = len(Ri & Ui) / len(Ri) if len(Ri) else 0

    # Compute Adherence
    adherence = all(info.get("fully_supported", False) for info in sentence_support_information)
    #adherence = 1 if all(info.get("fully_supported", False) for info in sentence_support_information) else 0
    
    return {
        "Context Relevance": context_relevance,
        "Context Utilization": context_utilization,
        "Completeness Score": completeness_score,
        "Adherence": adherence
    }

def get_metrics(attributes, total_sentences):
    if attributes.content:
        #print(attributes)
        result_content = attributes.content  # Access the content attribute
        # Extract the JSON part from the result_content
        json_start = result_content.find("{")
        json_end = result_content.rfind("}") + 1
        json_str = result_content[json_start:json_end]
        
        try:
            result_json = json.loads(json_str)
            # Compute metrics using the extracted attributes
            metrics = compute_metrics(result_json, total_sentences)
            print(metrics)
            return metrics        
        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError: {e}")