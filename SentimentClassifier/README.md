# Sentiment Classifier with BERT, RoBERTa, GPT, and MiniGPT

## Project Description

This project implements a sentiment classification system that predicts whether a movie review from the IMDB dataset is positive or negative. The system uses three pre-trained models: **BERT**, **RoBERTa**, and **OpenAI GPT** (with a custom MiniGPT architecture built using PyTorch).

The workflow includes:
1. **Data Preprocessing:** Downloading and preprocessing the IMDB dataset, splitting it into training, validation, and test datasets.
2. **Model Training:** Using publicly available weights for BERT, RoBERTa, and GPT, then fine-tuning these models for sentiment classification.
3. **Inference:** A simple chat application that allows customers to input movie reviews and receive predictions (positive/negative) from the trained models.

## Features

- **Sentiment Classification:** Classify movie reviews as positive or negative.
- **Model Training:** Fine-tune BERT, RoBERTa, and GPT models.
- **Custom MiniGPT:** A custom-built GPT architecture developed using PyTorch.
- **Interactive Chat App:** A frontend web application where users can interact with the models in real-time.

## Installation

### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.6+
- PyTorch
- Hugging Face Transformers
- Pandas
- Flask (for the chat app)

### Setting Up the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/KhawajaMuddassar/Large-Language-Models/sentiment-classifier.git

2. Navigate to the project directory:
   ```bash
   cd sentiment-classifier
3. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/macOS
   venv\Scripts\activate     # For Windows
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Running the Scripts
   
1. Running main.py for Model Training:
   ```bash
   python main.py
        or 
   python -m SentimentClassifier.main

2. Running app.py for the Chat Application:
   ```bash
   python app.py
        or 
   python -m SentimentClassifier.ClassifierApp.app

Once the server is running, you can access the chat app in your browser at http://127.0.0.1:5000/

### Chat App Features:
1. Users can input movie reviews.
2. The app will display the sentiment prediction (positive/negative) based on the trained models.

## Example Interaction (Screenshots)
Below is an example of the chat interface where users can interact with the models:

![Customer Sentiment Classifier](SentimentClassifier\ClassifierApp\static\images\app_.PNG)

# Video Demonstration
Here is a video showing the demo of how the chat app works and how users can interact with the sentiment classifier:
![Watch Demo Video](SentimentClassifier\ClassifierApp\static\images\SentimentClassifier.mp4)


