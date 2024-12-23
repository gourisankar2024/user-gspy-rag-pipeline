
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from generator.generate_metrics import generate_metrics
import logging

def compute_rmse_auc_roc_metrics(llm, dataset, vector_store, num_question):

     # Lists to accumulate ground truths and predictions for AUC-ROC computation
    all_ground_truth_relevance = []
    all_predicted_relevance = []

    all_ground_truth_utilization = []
    all_predicted_utilization = []

    all_ground_truth_adherence = []
    all_predicted_adherence = []

    # For each question in dataset get the metrics
    for i, document in enumerate(dataset):
        # Extract ground truth metrics from dataset
        ground_truth_relevance = dataset[i]['relevance_score']
        ground_truth_utilization = dataset[i]['utilization_score']
        ground_truth_adherence = 1 if dataset[i]['adherence_score'] else 0
        
        query = document['question']
        logging.info(f'Query number: {i + 1}')
        # Call the generate_metrics for each query
        metrics = generate_metrics(llm, vector_store, query)
        
        # Extract predicted metrics (ensure these are continuous if possible)
        predicted_relevance = metrics.get('Context Relevance', 0) if metrics else 0
        predicted_utilization = metrics.get('Context Utilization', 0) if metrics else 0
        predicted_adherence = 1 if metrics.get('Adherence', False) else 0
        
        # === Handle Continuous Inputs for RMSE ===
        all_ground_truth_relevance.append(ground_truth_relevance)
        all_predicted_relevance.append(predicted_relevance)
        all_ground_truth_utilization.append(ground_truth_utilization)
        all_predicted_utilization.append(predicted_utilization)
        
        all_ground_truth_adherence.append(ground_truth_adherence)
        all_predicted_adherence.append(predicted_adherence)

        if i == num_question:
          break
    
    # === Compute RMSE & AUC-ROC for the Entire Dataset ===
    try:
        relevance_rmse = root_mean_squared_error(all_ground_truth_relevance, all_predicted_relevance)
    except ValueError:
        relevance_rmse = None

    try:
        utilization_rmse = root_mean_squared_error(all_ground_truth_utilization, all_predicted_utilization)
    except ValueError:
        utilization_rmse = None

    try:
        print(f"All Ground Truth Adherence: {all_ground_truth_utilization}")
        print(f"All Predicted Utilization: {all_predicted_utilization}")
        adherence_auc = roc_auc_score(all_ground_truth_adherence, all_predicted_adherence)
    except ValueError:
        adherence_auc = None   

    print(f"Relevance RMSE score: {relevance_rmse}")
    print(f"Utilization RMSE score: {utilization_rmse}")
    print(f"Overall Adherence AUC-ROC: {adherence_auc}")
