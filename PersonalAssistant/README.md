# Intelligent Language Assistant with Preference-Tuned Responses

## Project overview

This project aims to develop an intelligent, instruction-following language assistant. The development process involves training a foundational model on curated instructional data, followed by refinement using Direct Preference Optimization (DPO) to align the model more closely with user preferences. For DPO, we extracted an instruction dataset from Ollama3, derived from the [Stanford Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json), enabling us to generate high-quality preference pairs for fine-tuning. The final model is deployed through an interactive app interface, allowing real-time engagement and feedback collection.

Due to limited computational resources, the base model was trained for only one epoch, which may result in occasional hallucinations or nonsensical outputs. Furthermore, DPO fine-tuning is sensitive and prone to training collapse, requiring careful handling of hyperparameters and preference data quality.

Despite these challenges, the model is integrated into a functional app that enables real-time interaction on real-world usage.

The workflow includes:
1. **Ollama Setup:** The setup process is handled automatically within 'main.py'. If Ollama is not detected on the local host, the script will automatically install it, make pull request for ollama3 and start the service.
2. **Data Collection:** Downloading large-scale instruction-following datasets, primarily Stanford Alpaca-based instruction dataset, adapted through Ollama3.preprocessing the IMDB dataset, 
3. **Preprocessing:** Deduplicated and standardized instruction-response pairs, tokenize using model-specific tokenizer and splitted it into training, validation, and test datasets, lastly processed batch loaders.
4. **Model Training:** Using publicly available weights for GPT, fine-tune using standard supervised learning for Cross-entropy on target tokens.
5. **DPO Model Training:** Direct Preference Optimization (DPO) is a method for aligning models directly with user preferences without requiring a reward model, an alternative to RLHF, so we fine-tuned Standalone Policy model along with reference and trained for (DPO Loss function)[https://arxiv.org/pdf/2305.18290] 
6. **User Interface:** A simple Desktop chat application that allows user to interact with model.

## Installation

### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.6+
- PyTorch
- Hugging Face Transformers
- threading
- tkinter (for the chat app)

### Setting Up the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/KhawajaMuddassar/Large-Language-Models/PersonalAssistant.git

2. Navigate to the project directory:
   ```bash
   cd PersonalAssistant
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
2. Running app.py for the Chat Application:
   ```bash
   python app.py
        or 
   python -m PersonalAssistant.app.app

## Example Interaction (Screenshots)
Below is an example of the chat interface where users can interact with the models:

![Intelligent Language Assistant](https://github.com/KhawajaMuddassar/Large-Language-Models/blob/main/PersonalAssistant/app/img/app.PNG)

# Video Demonstration
Here is a video showing the demo of how the chat app works and how users can interact with the sentiment classifier:
(<video controls src="https://github.com/KhawajaMuddassar/Large-Language-Models/blob/main/PersonalAssistant/app/img/Personal%20Assistant.mp4" title=".mp4"></video>)