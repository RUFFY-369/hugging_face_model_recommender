import gradio as gr
from huggingface_hub import list_models
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load sentence transformer model for similarity calculation
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to fetch models from Hugging Face based on dynamic task filter
def fetch_models_from_hf(task_filter, limit=10):
    models = list_models(filter=task_filter, limit=limit)
    model_data = [
        {
            "model_id": model.modelId,
            "tags": model.tags,
            "downloads": model.downloads,
            "likes": model.likes,
            "last_modified": model.lastModified
        }
        for model in models
    ]
    return model_data

# Function to normalize a list of values to a 0-1 range
def normalize(values):
    min_val, max_val = min(values), max(values)
    return [(v - min_val) / (max_val - min_val) if max_val > min_val else 0 for v in values]

# Function to get weighted recommendations based on task filter and additional metrics
def get_weighted_recommendations_from_hf(task_filter, weights=None):
    if weights is None:
        weights = {"similarity": 0.7, "downloads": 0.2, "likes": 0.1}

    model_data = fetch_models_from_hf(task_filter)

    if len(model_data) == 0:
        return "No models found for the specified task filter."

    model_ids = [model["model_id"] for model in model_data]
    model_tags = [' '.join(model["tags"]) for model in model_data]

    # Use a fixed user query based on task filter
    user_query = f"best model for {task_filter}"

    model_embeddings = semantic_model.encode(model_tags)
    user_embedding = semantic_model.encode(user_query)

    similarities = util.pytorch_cos_sim(user_embedding, model_embeddings)[0].numpy()

    downloads = normalize([model["downloads"] for model in model_data])
    likes = normalize([model["likes"] for model in model_data])

    final_scores = []
    for i in range(len(model_data)):
        score = (
            weights["similarity"] * similarities[i] +
            weights["downloads"] * downloads[i] +
            weights["likes"] * likes[i]
        )
        final_scores.append((model_ids[i], score, similarities[i], downloads[i], likes[i]))

    ranked_recommendations = sorted(final_scores, key=lambda x: x[1], reverse=True)

    result = []
    for rank, (model_id, final_score, sim, downloads, likes) in enumerate(ranked_recommendations, 1):
        result.append(f"Rank {rank}: Model ID: {model_id}")
    
    return '\n'.join(result)

# Gradio chatbot interface
def respond(task_filter, history=None, weights=None):
    # Provide model recommendations based on the task filter
    return get_weighted_recommendations_from_hf(task_filter, weights)

# Gradio Interface
demo = gr.Interface(
    fn=respond,
    inputs=[
        gr.Textbox(label="Task Filter", placeholder="Enter the task, e.g., text-classification, atari, question-answering"),
        gr.Textbox(value="You are using the Hugging Face model recommender system.", label="System message")
    ],
    outputs=gr.Textbox(label="Model Recommendations"),
    title="Hugging Face Model Recommender",
    description="This chatbot recommends models from Hugging Face based on the task or tag you're interested in. It combines various attributes of a model on hub like downloads, likes, etc. to suggest models with ranks from 1-10. In general term basically it intelligently combines the search filter for recommendation"
)

if __name__ == "__main__":
    demo.launch(share=True)
