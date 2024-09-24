# 💬 Model Recommendation Chatbot
This project is a Model Recommendation Chatbot designed to help users find the best machine learning models for specific tasks using the Hugging Face ecosystem. By leveraging the Gradio interface, the chatbot provides an interactive environment where users can ask questions about various models and datasets, and receive recommendations based on their needs.

## Key Features

- Interactive Chatbot: Utilizes a simple and intuitive interface powered by Gradio to engage users in conversation and recommend models from Hugging Face.
- Hugging Face Inference API: Integrates the Hugging Face Inference API to retrieve real-time model data and suggestions.
- Model Recommender: Helps users discover the most suitable models for tasks like text classification, image generation, translation, and more, using the huggingface_hub library.
- Gradio SDK v4.36.1: Built with the Gradio SDK for a streamlined UI and backend communication.

## Try it Out
This chatbot originally exists on Hugging Face Spaces. You can try the live version here: [Model Recommendation Chatbot on Hugging Face Spaces](https://huggingface.co/spaces/ruffy369/Model-Recommendation-Chatbot).

## Technology Stack

- Gradio: Front-end for the chatbot interface, enabling easy interaction with the recommendation system.
Hugging Face Hub: Integrates directly with Hugging Face’s extensive model hub for fetching model details and recommendations.
- Python: Primary language for developing the application.
Hugging Face Inference API: Provides the backbone for running model inferences and obtaining recommendations.
App Structure
- app.py: Main application file where the chatbot logic resides. It contains the chatbot workflow, user input handling, and model recommendations.
- huggingface_hub Integration: The app uses huggingface_hub to interact with the Hugging Face API, fetching relevant models based on user queries.

## Setup Instructions

- Clone this repository:
```
git clone https://github.com/RUFFY-369/hugging_face_model_recommender
```
- Install the required dependencies:
```
pip install -r requirements.txt
```
- Run the app locally:
```
python app.py
```

## License
This project is licensed under the MIT License.

