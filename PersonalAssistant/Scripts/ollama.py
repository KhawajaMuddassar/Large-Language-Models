import subprocess
import threading
import time
import platform
import urllib.request
import json
import time
import os
import gc
import psutil 
import random
from urllib.error import URLError, HTTPError
from tqdm import tqdm
from utils import format_input_to_alpaca
from get_data import GetData

def setup_and_run_ollama():
    # Install Dependencies (Platform-specific)
    system = platform.system()

    if system == "Linux":
        # For Linux (Ubuntu/Debian)
        subprocess.run(["sudo", "apt", "update"], check=True)
        subprocess.run(["sudo", "apt", "install", "-y", "pciutils"], check=True)
        subprocess.run(["curl", "-fsSL", "https://ollama.com/install.sh", "|", "sh"], check=True)
    elif system == "Windows":
        # For Windows
        try:
            subprocess.run(["winget", "install", "Ollama"], check=True)
        except subprocess.CalledProcessError:
            print("Error installing Ollama via winget. Please ensure winget is installed.")

    # Running Ollama
    def run_ollama_serve():
        if system == "Linux":
            subprocess.Popen(["ollama", "serve"])
        elif system == "Windows":
            subprocess.Popen(["ollama.exe", "serve"])  # Adjusted for Windows

    thread = threading.Thread(target=run_ollama_serve)
    thread.start()
    time.sleep(5)

    # Pulling Model
    if system == "Linux":
        subprocess.run(["ollama", "pull", "llama3"], check=True)
    elif system == "Windows":
        subprocess.run(["ollama.exe", "pull", "llama3"], check=True)

def is_ollama_running():    
    try:
        # Check Ollama running status
        result = subprocess.run(
            ["powershell", "-Command", "Get-Process | Where-Object {$_.Name -like 'ollama*'} | Select-Object -First 1"],
            capture_output=True, text=True
        )        
        if result.stdout.strip():
            print("Ollama3 is running.")
            return True
        else:
            print("Ollama3 is not running.")
            return False
    except Exception as e:
        print(f"Error checking Ollama3 status: {e}")
        return False

def run_query_chain(prompt, model='llama3', url='http://localhost:11434/api/chat'):
    """Run a sequence of query functions until a valid response is obtained"""
    
    # First, try QueryModelPost
    response = QueryModelPost(prompt, model, url)
    if response:        
        return response

    # Try QueryModelCrulStreaming    
    response = QueryModelCrulStreaming(prompt, model, url)
    if response:        
        return response
    return None 

def QueryModelPost(prompt, model='llama3', url='http://localhost:11434/api/chat'):        
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
        }
    } 
    payload = json.dumps(data).encode('utf-8')    
    request = urllib.request.Request(url, data=payload, method='POST')
    request.add_header('Content-Type', 'application/json')    
    response_data = ""
    try:
        with urllib.request.urlopen(request) as response:
            while True:
                line = response.readline().decode('utf-8')
                if not line:
                    break
                response_json = json.loads(line)
                response_data += response_json['message']['content']
    except HTTPError as e:
        print(f"HTTP error occurred: {e.code} {e.reason}")
    except URLError as e:
        print(f"URL error occurred: {e.reason}")
    except Exception as e:
        print(f"An error occurred: {e}")    
    return response_data

def QueryModelCrulStreaming(prompt, model="llama3", url="http://localhost:11434/api/chat"):    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
        }
    }
    json_data = json.dumps(data)
    curl_command = [
        "curl",
        "-X", "POST",
        url,
        "-H", "Content-Type: application/json",         
        "-d", json_data
    ]    
    response = subprocess.run(curl_command, capture_output=True, text=True)
    if response.returncode != 0:
        raise Exception(f"Error in cURL request: {response.stderr}")
    response_data = ""
    for line in response.stdout.splitlines():
        try:
            response_json = json.loads(line)
            response_data += response_json.get("message", {}).get("content", "")
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e} on line: {line}")    
    return response_data

def generate_model_responses(json_data):
    """Generates responses for the model and modifies the entries with politeness"""
    for i, entry in enumerate(tqdm(json_data, desc="Writing entries")):
        politeness = random.choice(["polite", "impolite"])    
        prompt = (
            f"Given the input `{format_input_to_alpaca(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"slightly rewrite the output to be more {politeness}."
            " Keep the modification minimal. Only return the generated response and nothing else."
        )
        response = run_query_chain(prompt)
        
        if politeness == "polite":
            json_data[i]["chosen"] = response
            json_data[i]["rejected"] = entry["output"]
        else:
            json_data[i]["rejected"] = response
            json_data[i]["chosen"] = entry["output"]

    return json_data  # Return modified json_data

def cleanup_ollama_processes():
    """Clean up Ollama-related processes and clear memory."""
    print("Performing Ollama-specific cleanup...")
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            if 'ollama' in proc.info['name'].lower():
                print(f"Terminating Ollama process with PID {proc.info['pid']}")
                proc.terminate()
                proc.wait()
        print("All Ollama processes terminated.")
    except Exception as e:
        print(f"Error during Ollama cleanup: {e}")
    
    gc.collect()
    print("Garbage collection completed.")

def extract_instruction(text):
    for content in text.split("\n"):
        if content:
            return content.strip()

if __name__ == '__main__':
    if is_ollama_running():
        pass
    else:
        setup_and_run_ollama()
        is_ollama_running()
        
    alpaca_file_dedup = 'PersonalAssistant/data/processed/alpaca_data_dedup.json'           
    ollama_file = "PersonalAssistant/data/raw/ollama_data.json"  
    if not os.path.exists(ollama_file):
        with open(ollama_file, 'w') as file:
            json.dump({}, file)
    ollama_file_dedup = "PersonalAssistant/data/processed/ollama_data_dedup.json"
    
    # Generate Ollama instructions
    with open(alpaca_file_dedup, "r") as file:
        instructions = json.load(file)[:2]
    model_res = generate_model_responses(instructions)
    with open(ollama_file, "w") as file:
        json.dump(model_res, file, indent=4)
        print("Data saved successfully.\n")
    with open(ollama_file, 'r') as file:
        data = json.load(file)

    # Deduplicate Ollama instructions
    deduplicated_data = GetData().remove_near_duplicates(data,remove_duplicates=True)
    
    with open(ollama_file_dedup, "w") as file:
            print(f'Writing Deduplicated file...')
            json.dump(deduplicated_data, file, indent=4)

    cleanup_ollama_processes()
    print("Memory cleanup completed.")