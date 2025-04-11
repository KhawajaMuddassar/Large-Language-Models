import subprocess
import sys

python_executable = sys.executable

# Run the get_data.py script
print(f'\nFeteching data for processing.....')
subprocess.run([python_executable, "PersonalAssistant/Scripts/get_data.py"], check=True)
print(f'Status: Complete')

# Run the ollama.py script
print(f'Generating user-preference dataset from Ollama3.....')
subprocess.run([python_executable, "PersonalAssistant/Scripts/ollama.py"], check=True)
print(f'Status: Complete\n')

# Run the gpt_model.py script
print(f'\nSetting Up GPT architecture.....')
subprocess.run([python_executable, "PersonalAssistant/Scripts/gpt_model.py"], check=True)
print(f'Status: Complete\n')

# Run the get_gpt.py script
subprocess.run([python_executable, "PersonalAssistant/Scripts/get_gpt.py"], check=True)

# Run the train_bert_hf.py script
print(f'Training fine tuned model.....\n')
subprocess.run([python_executable, "PersonalAssistant/Scripts/FT_train.py"], check=True)
print(f'Status: Complete\n')

# Run the train_dpo.py script
print(f'Training DPO model.....\n')
subprocess.run([python_executable, "PersonalAssistant/Scripts/train_dpo.py"], check=True)
print(f'Status: Complete\n')

print(f"Status: You can now run app.py to interact with model.\n")