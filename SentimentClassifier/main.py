import subprocess
import sys

python_executable = sys.executable

# Run the get_data.py script
print(f'\nFeteching data for processing.....\n')
subprocess.run([python_executable, "SentimentClassifier/Scripts/get_data.py"], check=True)
print(f'Status: Complete')

# Run the gpt_model.py script
print(f'\nSetting Up GPT architecture.....\n')
subprocess.run([python_executable, "SentimentClassifier/Scripts/gpt_model.py"], check=True)
print(f'Status: Complete\n')

# Run the gpt_train.py script
print(f'Downloading model weights and training model.....\n')
subprocess.run([python_executable, "SentimentClassifier/Scripts/gpt_train.py"], check=True)
print(f'\nStatus: Complete\n')

# Run the train_bert_hf.py script
print(f'Downloading model weights and training model.....\n')
subprocess.run([python_executable, "SentimentClassifier/Scripts/train_bert_hf.py"], check=True)
print(f'\nStatus: Complete\n')

# Run the baseline.py script
print(f'Baseline Compare.....\n')
subprocess.run([python_executable, "SentimentClassifier/Scripts/baseline.py"], check=True)
print(f'\nStatus: Complete\n')
print(f"Status: You can now run app.py for inference.\n")