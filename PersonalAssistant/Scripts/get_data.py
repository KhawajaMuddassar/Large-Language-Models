import requests
import os
from tqdm import tqdm
import pandas as pd
import json
import urllib.request
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class GetData:
    def __init__(self,):
        os.makedirs('PersonalAssistant/data/processed', exist_ok=True)
        os.makedirs('PersonalAssistant/data/raw', exist_ok=True)    
        self.alpaca_url = 'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json'    
        self.alpaca_file = 'PersonalAssistant/data/raw/alpaca_data.json'        
        self.alpaca_file_dedup = 'PersonalAssistant/data/processed/alpaca_data_dedup.json'
        
    def run(self,):
        self.get_alpaca_data()
        self.dedup()
        
    # Alpaca instruction data
    def get_alpaca_data(self):        
        if not os.path.exists(self.alpaca_file):
            with urllib.request.urlopen(self.alpaca_url) as res:
                text_data = res.read().decode('utf-8')
            with open(self.alpaca_file, 'w', encoding='utf-8') as file:
                file.write(text_data)
        else:
            with open(self.alpaca_file, 'r', encoding='utf-8') as file:
                text_data = file.read()
        with open(self.alpaca_file,'r') as file:
            data = json.load(file)
            print(f'\nDownload Complete.')            
        return data

    def preprocess_text(self, text):
        # Lowercase the text and handle empty or None values
        if not text:
            return ""
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def find_near_duplicates(self, json_data, threshold=0.9, key="instruction"):
        # Preprocess text and keep track of original indices
        preprocessed_data = [(i, self.preprocess_text(item.get(key, ""))) for i, item in enumerate(json_data)]
        preprocessed_data = [(i, txt) for i, txt in preprocessed_data if txt.strip()]

        if len(preprocessed_data) < 2:
            return json_data, [], 0  # Not enough valid entries to compare

        indices, text = zip(*preprocessed_data)

        # Vectorize the text data
        vectorizer = TfidfVectorizer(stop_words='english', analyzer='char', ngram_range=(1, 3))
        try:
            tfidf_matrix = vectorizer.fit_transform(text)
        except ValueError as e:
            print(f"TF-IDF Vectorizer Error: {e}")
            return json_data, [], 0

        # Compute cosine similarity between each pair of entries
        cos_sim_matrix = cosine_similarity(tfidf_matrix)

        near_duplicates = []
        indices_to_remove = set()

        for i in tqdm(range(len(cos_sim_matrix)), desc="Removing duplicates", ncols=100):
            for j in range(i + 1, len(cos_sim_matrix)):
                if cos_sim_matrix[i, j] > threshold:
                    idx_i = indices[i]
                    idx_j = indices[j]

                    if len(json_data[idx_i].get(key, "")) <= 1 or len(json_data[idx_j].get(key, "")) <= 1:
                        continue  # Skip if text is too short or empty
                    near_duplicates.append((json_data[idx_i], json_data[idx_j], cos_sim_matrix[i, j]))                
                    if key in ("input", "output"):  # Avoid removing based on instruction unless specified
                        indices_to_remove.add(idx_j)

        # Remove the near-duplicate entries
        filtered_json_data = [item for idx, item in enumerate(json_data) if idx not in indices_to_remove]
        duplicates_removed = len(indices_to_remove)

        return filtered_json_data, near_duplicates, duplicates_removed
    
    def remove_near_duplicates(self,json_data, remove_duplicates=False, threshold=0.9):
        
        total_duplicates_removed = 0          
        # Apply deduplication based on all keys that might contain text (like 'instruction', 'chosen', etc.)
        for key in json_data[0].keys():
            if remove_duplicates:
                json_data, near_duplicates, duplicates_removed = self.find_near_duplicates(json_data, key=key, threshold=threshold)
                total_duplicates_removed += duplicates_removed
            else:
                _, near_duplicates, duplicates_removed = self.find_near_duplicates(json_data, key=key, threshold=threshold)
                total_duplicates_removed += duplicates_removed        
        print(f'Total duplicates removed: {total_duplicates_removed}')
        return json_data
    
    def dedup(self,data):
        deduplicated_data = self.remove_near_duplicates(data,remove_duplicates=True)
        with open(self.alpaca_file_dedup, "w") as file:
            print(f'Writing Deduplicated file...')
            json.dump(deduplicated_data, file, indent=4)

if __name__ == '__main__':
    GetData().get_alpaca_data()
    alpaca_file = 'PersonalAssistant/data/raw/alpaca_data.json'
    with open(alpaca_file, 'r') as file:
            json_alpaca_full = json.load(file)[:1100]
            json_alpaca = json_alpaca_full
    print(f'Reading Data ...')                
    print(f'Deduplicating {len(json_alpaca)} of {len(json_alpaca_full)} rows...')
    GetData().dedup(json_alpaca)
