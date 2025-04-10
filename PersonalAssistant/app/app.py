import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PersonalAssistant')))
import tkinter as tk
import time
import threading
import random
import torch
import tiktoken
from tkinter import ttk
from pathlib import Path
import sys
import json
from Scripts.gpt_model import GPTModel
from Scripts.utils import (
    generate_text,
    text_to_token_id,
    token_to_text,
    format_input_to_alpaca)

class ChatbotApp:
    def __init__(self, root):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root = root
        self.root.title("Intelligent Language Assistant")
        window_width = 500
        window_height = 700
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        position_top = screen_height - window_height
        position_left = screen_width - window_width        
        self.root.geometry(f'{window_width}x{window_height}+{position_left}+{position_top}')
        self.root.config(bg='gray16')
        self.create_widgets()
        self.assistants = ["Hafsah Khawaja", "Hamnah Khawaja", "Hamnah Rizalde", "Hafsah Rizalde", "LuLu"]
        self.current_agent = random.choice(self.assistants)        
        self.loading_message = "Please wait, assigning assistant"
        self.loading_dots = 0
        self.loading_active = True  
        self.assistant_assigned = False         
        self.tokenizer, self.model, self.base_config = self.get_model_and_tokenizer()        
        self.start_chat()

    def create_widgets(self):        
        top_frame = tk.Frame(self.root, bg='gray16')
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        self.chat_box = tk.Text(self.root, height=30, font=('Arial', 12), width=60, wrap=tk.WORD, state=tk.DISABLED, bg='gray25', fg='SeaGreen3', bd=1, relief="flat")
        self.chat_box.pack(padx=10, pady=10)
        self.input_box = tk.Text(self.root, height=2, width=60, font=('Arial', 12), bg='gray25', fg='SeaGreen3', bd=1, relief="flat")
        self.input_box.pack(padx=10, pady=10)
        self.send_button = tk.Button(self.root, text="Send", command=self.send_message, bg="gray25", fg="SeaGreen3", font=('Arial', 12), 
                                     relief="flat", padx=10, pady=10)
        self.send_button.pack(pady= 10)         

    def append_message(self, message):        
        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.insert(tk.END, f"{message}\n")
        self.chat_box.config(state=tk.DISABLED)
        self.chat_box.yview(tk.END)  

    def start_chat(self):        
        self.append_message(self.loading_message + "......")       
        self.loading_effect()        
        self.root.after(5000, self.assign_assistant)

    def loading_effect(self):        
        if self.loading_active:            
            self.chat_box.config(state=tk.NORMAL)
            self.chat_box.delete("1.0", tk.END)  
            self.chat_box.insert(tk.END, self.loading_message + "." * self.loading_dots)  
            self.chat_box.config(state=tk.DISABLED)            
            if self.loading_dots < 3:
                self.loading_dots += 1
            else:
                self.loading_dots = 0            
            if self.loading_active:
                self.loading_timer = self.root.after(500, self.loading_effect)

    def stop_loading_effect(self):        
        if hasattr(self, 'loading_timer'):
            self.root.after_cancel(self.loading_timer)

    def assign_assistant(self):        
        if not self.assistant_assigned:            
            self.stop_loading_effect()            
            self.append_message(f'\nHello! I am {self.current_agent} here to assist you.\n')            
            self.assistant_assigned = True

    def send_message(self):        
        user_input = self.input_box.get("1.0", tk.END).strip()
        if not user_input:
            return        
        self.append_message(f"User:\n{user_input}")
        self.input_box.delete("1.0", tk.END)        
        self.pulsating_typing_message()        
        threading.Thread(target=self.get_response_from_model, args=(user_input, self.current_agent)).start()

    def pulsating_typing_message(self):        
        self.typing_dots = 0  
        self.typing_active = True
        self.update_typing_message()

    def update_typing_message(self):        
        if self.typing_active:            
            self.chat_box.config(state=tk.NORMAL)
            current_chat = self.chat_box.get("1.0", tk.END)
            if f"{self.current_agent}: typing ..." in current_chat:
                current_chat = current_chat.rsplit(f"{self.current_agent}: typing ...", 1)[0]
            self.chat_box.delete("1.0", tk.END)
            self.chat_box.insert(tk.END, current_chat)
            self.chat_box.insert(tk.END, f"{self.current_agent}: typing {'.' * self.typing_dots}...")            
            if self.typing_dots < 3:
                self.typing_dots += 1
            else:
                self.typing_dots = 0            
            if self.typing_active:
                self.root.after(500, self.update_typing_message)

    def stop_typing_message(self):        
        self.typing_active = False
        self.clear_typing_message()

    def clear_typing_message(self):        
        current_chat = self.chat_box.get("1.0", tk.END)
        if f"{self.current_agent}: typing ..." in current_chat:
            new_chat = current_chat.rsplit(f"{self.current_agent}: typing ...", 1)[0]
            self.chat_box.config(state=tk.NORMAL)
            self.chat_box.delete("1.0", tk.END)
            self.chat_box.insert(tk.END, new_chat)
            self.chat_box.config(state=tk.DISABLED)

    def get_model_and_tokenizer(self):
        with open('PersonalAssistant/model_config.json') as file:
            config = json.load(file)
        tokenizer = tiktoken.get_encoding("gpt2")
        config['base_config'].update(config['model_config']['gpt2-medium'])
        model_path = Path("PersonalAssistant/model/modelfiles/dpo_Model.pth")         
        model = GPTModel(config['base_config'])                      
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device) )
        model.eval()        
        return tokenizer, model, config['base_config']
    
    def get_response_from_model(self, message, agent):                
        prompt = f"""### Instruction:{message} """   
        token_ids = generate_text(
            model=self.model,
            idx=text_to_token_id(message, self.tokenizer).to(self.device),
            max_new_tokens=150,
            context_size=self.base_config["context_length"],
            eos_id=50256
        )
        text = token_to_text(token_ids, self.tokenizer)             
        response = self.extract_response(text, prompt)
        if not response:
            response = "Sorry, I couldn't generate a response."
        self.stop_typing_message()
        self.root.after(0, lambda: self.append_message(f"{agent}:\n {response}"))

    def extract_response(self, response_text, input_text):        
        return response_text[len(input_text):].replace("### Response:", "").strip() 
        
        

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()