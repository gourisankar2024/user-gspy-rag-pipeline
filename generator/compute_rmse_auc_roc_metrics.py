
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from generator.compute_metrics import get_metrics
from generator.extract_attributes import extract_attributes
from generator.generate_response import generate_response
from retriever.retrieve_documents import retrieve_top_k_documents

def compute_rmse_auc_roc_metrics(llm, dataset, vector_store):

     # Lists to accumulate ground truths and predictions for AUC-ROC computation
    all_ground_truth_relevance = []
    all_predicted_relevance = []

    all_ground_truth_utilization = []
    all_predicted_utilization = []

    all_ground_truth_adherence = []
    all_predicted_adherence = []

    # To store RMSE scores for each question
    relevance_scores = []
    utilization_scores = []
    adherence_scores = []

    for i, sample in enumerate(dataset):
        print(sample)
        sample_question = sample['question']
        
        # Extract ground truth metrics from dataset
        ground_truth_relevance = dataset[i]['relevance_score']
        ground_truth_utilization = dataset[i]['utilization_score']
        ground_truth_completeness = dataset[i]['completeness_score']
        
        # Step 1: Retrieve relevant documents
        relevant_docs = retrieve_top_k_documents(vector_store, sample_question, top_k=5)

        # Step 2: Generate a response using LLM
        response, source_docs = generate_response(llm, vector_store, sample_question, relevant_docs)

        # Step 3: Extract attributes
        attributes, total_sentences = extract_attributes(sample_question, source_docs, response)
    
        # Call the process_attributes method in the main block
        metrics = get_metrics(attributes, total_sentences)
        
        # Extract predicted metrics (ensure these are continuous if possible)
        predicted_relevance = metrics['Context Relevance']
        predicted_utilization = metrics['Context Utilization']
        predicted_completeness = metrics['Completeness Score']
        
        # === Handle Continuous Inputs for RMSE ===
        relevance_rmse = root_mean_squared_error([ground_truth_relevance], [predicted_relevance])
        utilization_rmse = root_mean_squared_error([ground_truth_utilization], [predicted_utilization])
        #adherence_rmse = mean_squared_error([ground_truth_adherence], [predicted_adherence], squared=False)
        
        # === Handle Binary Conversion for AUC-ROC ===
        binary_ground_truth_relevance = 1 if ground_truth_relevance > 0.5 else 0
        binary_predicted_relevance = 1 if predicted_relevance > 0.5 else 0

        binary_ground_truth_utilization = 1 if ground_truth_utilization > 0.5 else 0
        binary_predicted_utilization = 1 if predicted_utilization > 0.5 else 0

        #binary_ground_truth_adherence = 1 if ground_truth_adherence > 0.5 else 0
        #binary_predicted_adherence = 1 if predicted_adherence > 0.5 else 0
        
        # === Accumulate data for overall AUC-ROC computation ===
        all_ground_truth_relevance.append(binary_ground_truth_relevance)
        all_predicted_relevance.append(predicted_relevance)  # Use probability-based predictions

        all_ground_truth_utilization.append(binary_ground_truth_utilization)
        all_predicted_utilization.append(predicted_utilization)

        #all_ground_truth_adherence.append(binary_ground_truth_adherence)
        #all_predicted_adherence.append(predicted_adherence)

        # Store RMSE scores for each question
        relevance_scores.append(relevance_rmse)
        utilization_scores.append(utilization_rmse)
        #adherence_scores.append(adherence_rmse)
        if i == 9:  # Stop after processing the first 10 rows
          break
    # === Compute AUC-ROC for the Entire Dataset ===
    try:
        print(f"All Ground Truth Relevance: {all_ground_truth_relevance}")
        print(f"All Predicted Relevance: {all_predicted_relevance}")
        relevance_auc = roc_auc_score(all_ground_truth_relevance, all_predicted_relevance)
    except ValueError:
        relevance_auc = None

    try:
        print(f"All Ground Truth Utilization: {all_ground_truth_utilization}")
        print(f"All Predicted Utilization: {all_predicted_utilization}")
        utilization_auc = roc_auc_score(all_ground_truth_utilization, all_predicted_utilization)
    except ValueError:
        utilization_auc = None   
        
    print(f"Relevance RMSE (per question): {relevance_scores}")
    print(f"Utilization RMSE (per question): {utilization_scores}")
    #print(f"Adherence RMSE (per question): {adherence_scores}")
    print(f"\nOverall Relevance AUC-ROC: {relevance_auc}")
    print(f"Overall Utilization AUC-ROC: {utilization_auc}")
