import gradio as gr
import logging
from generator.compute_rmse_auc_roc_metrics import compute_rmse_auc_roc_metrics

def launch_gradio(vector_store, dataset, gen_llm, val_llm):
    """
    Launch the Gradio app with pre-initialized objects.
    """
    def answer_question_with_metrics(query):
        try:
            logging.info(f"Processing query: {query}")
            
            # Generate metrics using the passed objects
            from main import generate_metrics
            response, metrics = generate_metrics(gen_llm, val_llm, vector_store, query, 1)
            
            response_text = f"Response: {response}\n\n"
            metrics_text = "Metrics:\n"
            for key, value in metrics.items():
                if key != 'response':
                    metrics_text += f"{key}: {value}\n"
            
            return response_text, metrics_text
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return f"An error occurred: {e}"

    def compute_and_display_metrics():
        try:
            # Call the function to compute metrics
            relevance_rmse, utilization_rmse, adherence_auc = compute_rmse_auc_roc_metrics(
                gen_llm, val_llm, dataset, vector_store, 10
            )
            
            # Format the result for display
            result = (
                f"Relevance RMSE Score: {relevance_rmse}\n"
                f"Utilization RMSE Score: {utilization_rmse}\n"
                f"Overall Adherence AUC-ROC: {adherence_auc}\n"
            )
            return result
        except Exception as e:
            logging.error(f"Error during metrics computation: {e}")
            return f"An error occurred: {e}"

    # Define Gradio Blocks layout
    with gr.Blocks() as interface:
        interface.title = "Real Time RAG Pipeline Q&A"
        gr.Markdown("### Real Time RAG Pipeline Q&A")  # Heading
        gr.Markdown("Ask a question and get a response with metrics calculated from the RAG pipeline.")  # Description
        
        with gr.Row():
            query_input = gr.Textbox(label="Ask a question", placeholder="Type your query here")
        with gr.Row():
            clear_query_button = gr.Button("Clear")  # Clear button
            submit_button = gr.Button("Submit", variant="primary") # Submit button
        with gr.Row():
            answer_output = gr.Textbox(label="Response", placeholder="Response will appear here")
        with gr.Row():
            metrics_output = gr.Textbox(label="Metrics", placeholder="Metrics will appear here")
        with gr.Row():
            compute_rmse_button = gr.Button("Compute RMSE & AU-ROC", variant="primary")
            rmse_output = gr.Textbox(label="RMSE & AU-ROC Score", placeholder="RMSE & AU-ROC score will appear here")
    
        
        # Define button actions
        submit_button.click(fn=answer_question_with_metrics, inputs=[query_input], outputs=[answer_output, metrics_output])
        clear_query_button.click(fn=lambda: "", outputs=[query_input])  # Clear query input 
        compute_rmse_button.click(fn=compute_and_display_metrics, outputs=[rmse_output])

    interface.launch()
