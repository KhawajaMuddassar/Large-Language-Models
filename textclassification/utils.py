import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))
import torch
import urllib.request
from datetime import datetime
import zipfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import tiktoken

#Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Tokenizer
tokenizer = tiktoken.get_encoding('gpt2')

def fetch_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:            
            out_file.write(response.read())
            
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)
    
    # add tsv extention
    original_file = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file, data_file_path)
    print(f"Dataset downloaded and extracted to {data_file_path}")

# Handle class Imbalance (RandomOverSampling)Fine-Tuning-Text-Classification\data\SMSSpamCollection\SMSSpamCollection.tsv
def balance_dataset_strategy():
    """
    Balances a dataset by automatically applying either oversampling or undersampling
    based on the class distribution in the 'Label' column.
    - If the minority class has fewer instances, oversampling is applied to it.
    - If the majority class has more instances, undersampling is applied to it.
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame with features and a 'Label' column. The 'Label' column
        should contain categorical values such as 'spam' and 'ham'.
    Returns:
    --------
    pd.DataFrame
        A new DataFrame with balanced class distributions.
    """
    # Download and unzip the dataset
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = os.path.join("textclassification/data","SMSSpamCollection")
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv" 
    fetch_data(url, zip_path, extracted_path, data_file_path)   
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])                      
    
    class_counts = df["Label"].value_counts()
    num_spam = class_counts.get("spam", 0)
    num_ham = class_counts.get("ham", 0)
    print(f"Original class distribution: {class_counts}")    
    
    if num_spam > num_ham:
        # "ham" is the minority class
        minority_class = "ham"
        majority_class = "spam"
        minority_count = num_ham
        majority_count = num_spam
    else:
        # "spam" is the minority class
        minority_class = "spam"
        majority_class = "ham"
        minority_count = num_spam
        majority_count = num_ham
        
    X = df.drop('Label', axis=1)
    y = df['Label']        
    
    if minority_count >= majority_count:
        # Random Under Sampling
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
        print(f"Undersampling applied to the '{majority_class}' class in the 'Label' column.")
        print(f"Resampled dataset shape (after undersampling): {Counter(y_res)}")
    elif minority_count < majority_count:
        # Random Over Sampling
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)    
        print(f"Oversampling applied to the '{minority_class}' class in the 'Label' column.")
        print(f"Resampled dataset shape (after oversampling): {Counter(y_res)}")
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plot_filename = f"textclassification/plots/class_distribution_{timestamp}.png"
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)    
    # Plot the resampled class distribution
    resampled_class_counts = pd.Series(y_res).value_counts()
    plot_class_distribution(class_counts, resampled_class_counts, plot_filename)    
    balanced_df = pd.DataFrame(X_res, columns=X.columns)
    balanced_df['Label'] = y_res  
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})       
    return balanced_df

def plot_class_distribution(original_counts: pd.Series, resampled_counts: pd.Series, plot_filename: str):
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6)) 
    # Original Class Distribution
    bars1 = ax[0].bar(original_counts.index, original_counts.values, color=['blue', 'orange'], edgecolor='black')
    ax[0].set_title('Original Class Distribution')
    ax[0].set_xlabel('Class')
    ax[0].set_ylabel('Number of Instances')
    for bar in bars1:
        height = bar.get_height()  # Get the height count of each bar
        ax[0].text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom', fontweight='bold')
    # Resampled Class Distribution
    bars2 = ax[1].bar(resampled_counts.index, resampled_counts.values, color=['blue', 'orange'], edgecolor='black')
    ax[1].set_title('Resampled Class Distribution')
    ax[1].set_xlabel('Class')
    ax[1].set_ylabel('Number of Instances')
    for bar in bars2:
        height = bar.get_height()  # Get the height count of each bar
        ax[1].text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close() 

# Train, Test, and validaiton dataset 
def split_data(train_split, val_split):   
    df = balance_dataset_strategy() 
    # Shuffle 
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)    
    #Split Indices 
    train = int(len(df) * train_split)
    val = train + int(len(df) * val_split)    
    #Split Data
    train_df = df[:train]
    val_df = df[train:val]
    test_df = df[val:]
    return train_df, val_df, test_df

    
    
# Test 
#train_df, val_df, test_df = split_data( 0.7, 0.1)
#print(len(train_df))
#print(len(val_df))
#print(len(test_df))