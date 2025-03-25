import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))

import pandas as pd
import tarfile
import urllib.request
import time
import shutil
from tqdm import tqdm

class get_data:
    def __init__(self):
        pass
    
    def run(self):
        self.download_and_extract()
        print("Creating data frames ...")
        df = self.load_dataset_to_dataframe()
        print("Partitioning and saving data frames ...")
        self.partition_and_save(df)
        
    # Callback function for real-time feedback on file transfer
    def reporthook(
        self,
        count,
        block_size,
        total_size
        ):
        global start_time
        if count == 0:
            start_time = time.time()
        else:
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            percent = count * block_size * 100 / total_size
            speed = int(progress_size / (1024 * duration)) if duration else 0
            sys.stdout.write(
                f"\r{int(percent)}% | {progress_size / (1024**2):.2f} MB "
                f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
            )
            sys.stdout.flush()
            
    def extraction_progress_hook(
        self,
        tar_file,
        path
        ):
        total_members = len(tar_file.getmembers())
        current_member = 0
        for member in tar_file:
            current_member += 1
            tar_file.extract(member, path)
            progress = (current_member / total_members) * 100
            print(f"Extracting: {progress:.2f}% - {current_member}/{total_members} files", end='\r')
        print("Extraction complete.")
        
    def download_and_extract(
        self,
        dataset_url ="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        target_file = "aclImdb_v1.tar.gz",
        directory = "SentimentClassifier/data"        
        ):
        zipped_dir = os.path.join(directory, "zipped")
        if not os.path.exists(zipped_dir):
            os.makedirs(zipped_dir)
    
        unzipped_dir = os.path.join(directory, "unzipped")
        if os.path.exists(unzipped_dir): 
            print(f"Directory `{unzipped_dir}` already exists. Removing it...")
            shutil.rmtree(unzipped_dir)  
   
        os.makedirs(unzipped_dir)
        target_file_path = os.path.join(zipped_dir, target_file)

        # Download the dataset and extract it
        print(f"Downloading dataset from {dataset_url} ...")
        urllib.request.urlretrieve(dataset_url, target_file_path, self.reporthook)
        print(f"\nDownload complete. Extracting dataset to {unzipped_dir} ...")

        # Extract the tar file to "unzipped" folder
        with tarfile.open(target_file_path, "r:gz") as tar:
            self.extraction_progress_hook(tar, unzipped_dir)                    

    # Load dataset into Pandas Dataframe
    def load_dataset_to_dataframe(
        self,
        basepath='SentimentClassifier/data/unzipped/aclImdb', 
        labels={"pos":1, "neg":0}
        ):
        data_frame = [] # Store chunks of dataframe
        for subset in ("test", "train"):
            for label in ("pos",'neg'):
                path = os.path.join(basepath, subset, label)
                files = sorted(os.listdir(path))
                for file in tqdm(files, desc=f"Processing {subset}/{label}", unit="file"):
                    file_path = os.path.join(path, file)                
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Create dataframe for each file and add it to the list
                        data_frame.append(pd.DataFrame({'text':[f.read()], 'label':[labels[label]]}))
        # Concatenate all Dataframe togather
        df = pd.concat(data_frame, ignore_index=True)
        df = df.sample(frac=1, random_state=123).reset_index(drop=True) # Shuffle the dataframe
        return df 

     # Partition Data and save it to disk
    def partition_and_save(
        self,
        df,
        sizes=(35000, 5000, 10000),
        loc = 'SentimentClassifier/data/processed'
        ):    
        # Reshuffle Dataframe
        df_shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)
    
        # Get indices for each partition
        train_end = sizes[0]
        val_end = sizes[0] + sizes[1]
        test_end = sizes[0] + sizes[1] + sizes[2]
    
        # Split the dataframe
        train = df_shuffled.iloc[:train_end]
        val = df_shuffled.iloc[train_end:val_end]
        test = df_shuffled.iloc[val_end:test_end]
    
        # Save to CSVs
        train.to_csv('SentimentClassifier/data/processed/train.csv', index=False)
        val.to_csv('SentimentClassifier/data/processed/val.csv', index=False)
        test.to_csv('SentimentClassifier/data/processed/test.csv', index=False)
        print(f'Data is partitioned and saved to {loc} directory.\n')
        
if __name__ == "__main__":
    get_data().run()    