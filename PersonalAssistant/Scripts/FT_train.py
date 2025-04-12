import torch
import re
import os
import json
import time
from process_data import ProcessData
from get_gpt import get_gpt
from gpt_model import GPTModel
from utils import (load_weights_into_gpt,
                   train_model_simple,
                   format_input_to_alpaca,
                   tokenizer,
                   plot_losses)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read Config Files
with open ('ChatAssistant/model_config.json','r') as file:
    config = json.load(file)
with open ('ChatAssistant/data/processed/alpaca_data_dedup.json', 'r') as file:
    data = json.load(file)#[:1100]    
    
train_loader, val_loader, val, test = ProcessData(data).GetDataReady()

#Update config based on choosen model
choose_model = str(input('Choose Model size (gpt2-small, gpt2-medium, gpt2-large, gpt2-xl):'))
config['base_config'].update(config['model_config'][choose_model])

setting, params = get_gpt().download_and_load_gpt(choose_model = choose_model )
model = GPTModel(config['base_config'])
load_weights_into_gpt(model, params)
model.eval()
model.to(device)

# Train Model
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 2
train_losses, val_losses, token_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context=format_input_to_alpaca(val[0]),
    tokenizer=tokenizer)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# Save Model
os.makedirs('PersonalAssistant/model/modelfiles',exist_ok=True)
torch.save(model.state_dict(), 'PersonalAssistant/model/modelfiles/FT_Model.pth')
print(f"FT Model saved ")

#Plot losses
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, token_seen, train_losses, val_losses)

