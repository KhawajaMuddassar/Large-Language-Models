import torch 
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class DatasetGPT(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_len,        
        stride
        ) -> None:        
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(data, allowed_special={'<|endoftext|>'})
        for i in range(0, len(token_ids) - max_len, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_len]))
            self.target_ids.append(torch.tensor(token_ids[i + 1:i + max_len + 1]))                                                                                                                                          
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size=50257, out_dim=768, context_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.out_dim = out_dim
        self.context_size = context_size
        with open('data/raw/the-text.txt', 'r') as f:
            data = f.read()
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.downloader = dataloader(self.tokenizer, data, max_len=1024, stride=4)  
    def EmbeddingLayers(self, batch):
        token_embedding_layer = nn.Embedding(self.vocab_size, self.out_dim)
        pos_embedding_layer = nn.Embedding(self.context_size, self.out_dim)
        for batch in self.downloader:
            input_ids, target_ids = batch
            token_embed = token_embedding_layer(input_ids)
            pos_embed = pos_embedding_layer(torch.arange(self.context_size))
            input_embed = token_embed + pos_embed
            return input_embed

def dataloader(    
    data,
    max_len=256,
    stride=128,
    batch_size=4,
    num_workers=0,
    shuffle=True, 
    drop_last=True,
    ):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = DatasetGPT(data,tokenizer, max_len,  stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)


