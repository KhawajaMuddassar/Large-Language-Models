import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Large-Language-Models')))
import torch
from flask import Flask, request, jsonify, render_template
from pathlib import Path
import tiktoken
import json
from SentimentClassifier.Scripts.gpt_model import GPTModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_cors import CORS


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Serve the chat.html file at the root URL
@app.route('/')
def index():
    return render_template('chat.html')  # Renders the chat.html located in 'templates' folder

# Define a function to load models
def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open('SentimentClassifier/ClassifierApp/model_config.json','r') as c:
        cfg = json.load(c)
    #cfg['base_config'].update(cfg['model_config']['gpt2-small'])
    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }
    BASE_CONFIG.update(cfg['model_config']['gpt2-small'])
    if model_name == "GPT2-Classifier":
        model_path = Path("SentimentClassifier/model/gpt_model/gptclassifier.pth") 
        if not model_path.exists():
            print(f"Could not find the {model_path} file.Please run 'SentimentClassifier\main.py'")
        tokenizer = tiktoken.get_encoding('gpt2')
        model = GPTModel(BASE_CONFIG)
        num_classes = 2
        model.out_head = torch.nn.Linear(in_features=BASE_CONFIG['emb_dim'], out_features=num_classes)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
    elif model_name == "Roberta-Classifier":
        model_path = Path("SentimentClassifier/model/bert_model/robertaclassifier.pth")
        if not model_path.exists():
            print(f"Could not find the {model_path} file.Please run 'SentimentClassifier\main.py'")
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
        model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-large", num_labels=2)
        model.classifier.out_proj = torch.nn.Linear(in_features=1024, out_features=2)
        checkpoint = torch.load(model_path,map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
    elif model_name == "BERT-Classifier":
        model_path = Path("SentimentClassifier/model/bert_model/distilbertclassifier.pth")
        if not model_path.exists():
            print(f"Could not find the {model_path} file.Please run 'SentimentClassifier\main.py'")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2 )
        model.out_head = torch.nn.Linear(in_features=768, out_features=2)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
    
    return model, tokenizer

# Endpoint to handle the chat
@app.route('/get_response', methods=['POST'])
def get_response():
    
    data = request.get_json()
    model_name = data.get('model')
    message = data.get('message')

    # Load the selected model
    model, tokenizer = load_model(model_name)
    
    # Tokenize the input message and get the model's output
    tokens = tokenizer.encode(message)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # If the model is GPT-2 based (like the GPT-2 Classifier), use the model's forward pass
    if model_name == "GPT2-Classifier":
        with torch.no_grad():
            output = model(inputs)
            print(f"GPT2 output: {output}")  # Print the output to inspect it

            # Check the output shape
            if isinstance(output, tuple):
                logits = output[0]  # In case it's a tuple
            else:
                logits = output

            print(f"Logits shape: {logits.shape}")

            # Get predicted class for each token (0 or 1) from the logits
            token_predictions = torch.argmax(logits, dim=-1)  # Apply argmax on the last dimension (class dimension)
            print(f"Token-level predictions: {token_predictions}")

            # Aggregate predictions (here, using majority vote or averaging)
            # If you want to take the majority class, we can use mode
            predicted_class = torch.mode(token_predictions.flatten()).values.item()  # Majority vote across all tokens

            response_mapping = {0: "Negative sentiment", 1: "Positive sentiment"}
            response_text = response_mapping.get(predicted_class, "Unknown sentiment")

    # If the model is BERT-based (like RoBERTa or DistilBERT), process similarly
    elif model_name in ["Roberta-Classifier", "BERT-Classifier"]:
        with torch.no_grad():
            outputs = model(inputs)
            logits = outputs.logits  # The logits (raw predictions) from the model
            prediction = torch.argmax(logits, dim=-1).item()  # Predicted class (0 or 1)
            response_mapping = {0: "Negative sentiment", 1: "Positive sentiment"}
            response_text = response_mapping.get(prediction, "Unknown sentiment")
    
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True)
