def compute_metrics(attributes, total_sentences):
    # Extract relevant information from attributes
    all_relevant_sentence_keys = attributes.get("all_relevant_sentence_keys", [])
    all_utilized_sentence_keys = attributes.get("all_utilized_sentence_keys", [])
    sentence_support_information = attributes.get("sentence_support_information", [])

    # Compute Context Relevance
    context_relevance = len(all_relevant_sentence_keys) / total_sentences if total_sentences else 0
    
    # Compute Context Utilization
    context_utilization = len(all_utilized_sentence_keys) / len(sentence_support_information) if sentence_support_information else 0
    
    # Compute Completeness
    completeness = all(info.get("fully_supported", False) for info in sentence_support_information)
    
    # Compute Adherence
    adherence = attributes.get("overall_supported", False)
    
    return {
        "Context Relevance": context_relevance,
        "Context Utilization": context_utilization,
        "Completeness": completeness,
        "Adherence": adherence
    }