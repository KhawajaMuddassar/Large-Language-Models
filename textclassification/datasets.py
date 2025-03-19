import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))
import torch
import torch.nn as nn
import pandas as pd
from textclassification.utils import tokenizer, device, split_data
from torch.utils.data import DataLoader, Dataset

class EmailDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=None, pad_token_ids=50256):
        self.data = data
        self.encoded_texts = [ tokenizer.encode(text) for text in self.data['Text']]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequence if long than max_length
            self.encoded_texts = [ encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
    
        # Pad sequences to longest sequences
        self.encoded_texts = [
            encoded_text + [pad_token_ids] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]['Label']
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        return max([len(encoded_text) for encoded_text in self.encoded_texts])

class DatasetLoader:
    def __init__(self, batch_size=8, num_worker=0):
        self.batch_size = batch_size        
        self.num_workers = num_worker
    def get_dataloader(self):
        train_df, val_df, test_df = split_data( 0.7, 0.1) 
        train_dataset = EmailDataset(train_df,tokenizer,max_length=None)
        val_dataset = EmailDataset(val_df, tokenizer,max_length=train_dataset.max_length)
        test_dataset = EmailDataset(test_df, tokenizer,max_length=train_dataset.max_length)
        
        train_dl = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )
        val_dl = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )
        test_dl = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )
        return train_dataset,val_dataset,test_dataset, train_dl, val_dl, test_dl
