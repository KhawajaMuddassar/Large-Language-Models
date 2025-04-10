import torch
import tiktoken
from functools import partial
from torch.utils.data import DataLoader, Dataset
from utils import format_input_to_alpaca, collate_fn


class ProcessData:
    def __init__(self,data):
        self.data = data
        self.tokenizer = tiktoken.get_encoding('gpt2')

    def GetDataReady(self):

        cstm_collate_fn = partial(
            collate_fn,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            allowed_max_length=256)

        # Split Data
        train_portion = int(len(self.data) * 0.85)  # 85% for training
        test_portion = int(len(self.data) * 0.1)    # 10% for testing
        val_portion = len(self.data) - train_portion - test_portion
        train = self.data[:train_portion]
        test = self.data[train_portion:train_portion + test_portion]
        val = self.data[train_portion + test_portion:]

        # Data Loaders
        train_dataset = InstructionDataset(train, self.tokenizer)
        val_dataset = InstructionDataset(val, self.tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            num_workers=0,
            collate_fn=cstm_collate_fn,
            shuffle=True,
            drop_last=True
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,
            drop_last=False,
            collate_fn=cstm_collate_fn,
            num_workers=0
        )
        print(f'Total rows: {len(self.data)}')
        print(f'Train Dataset: {len(train)}')
        print(f'Test Dataset: {len(test)}')
        print(f'Val Dataset: {len(val)}')
        return train_loader, val_loader, val, test

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        #pre-tokenize text
        self.encoded_text = []
        for entry in data:
            instruction_input = format_input_to_alpaca(entry)
            response = f"\n\nResponse:\n{entry['output']}"
            text = instruction_input + response
            self.encoded_text.append(tokenizer.encode(text))
    def __getitem__(self, index):
        return self.encoded_text[index]
    def __len__(self,): return len(self.data)

########################## DPO Train ###########################

class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.encoded_text = []
        for entry in data:
            prompt = format_input_to_alpaca(entry)
            rejected_response = entry['rejected']
            chosen_response = entry['chosen']
            prompt_tokens = tokenizer.encode(prompt)
            chosen_text = f'{prompt}\n\n### Response:\n{chosen_response}'
            rejected_text = f'{prompt}\n\n### Response:\n{rejected_response}'
            chosen_tokens = tokenizer.encode(chosen_text)
            rejected_tokens = tokenizer.encode(rejected_text)
            self.encoded_text.append(
                {"prompt":prompt_tokens,
                "chosen":chosen_tokens,
                "rejected":rejected_tokens,}
                )
    def __getitem__(self, index):
        return self.encoded_text[index]

    def __len__(self,): return len(self.data)

class ProcessDataDPO:
    def __init__(self,data):
        self.data = data
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def GetDataReady(self):

        cstm_collate_fn = partial(
            self.collate_fn,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            mask_prompt_tokens=True,
            allowed_max_length=256
            )

        # Split Data
        train_portion = int(len(self.data) * 0.85)  # 85% for training
        test_portion = int(len(self.data) * 0.1)    # 10% for testing
        val_portion = len(self.data) - train_portion - test_portion
        train = self.data[:train_portion]
        test = self.data[train_portion:train_portion + test_portion]
        val = self.data[train_portion + test_portion:]

        # Data Loaders
        train_dataset = PreferenceDataset(train, self.tokenizer)
        val_dataset = PreferenceDataset(val, self.tokenizer)
        test_dataset = PreferenceDataset(test, self.tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            num_workers=0,
            collate_fn=cstm_collate_fn,
            shuffle=True,
            drop_last=True
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,
            drop_last=False,
            collate_fn=cstm_collate_fn,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            drop_last=False,
            collate_fn=cstm_collate_fn,
            num_workers=0
        )
        return train_loader, val_loader, val, test

    def collate_fn(self,batch, pad_token_id=50256, allowed_max_length=None, mask_prompt_tokens=True, device='cpu'):
        # Initialize the lists to hold batch data
        batch_data ={
            "prompt":[],
            "chosen":[],
            "rejected":[],
            "chosen_mask":[],
            "rejected_mask":[]
            }
        # get longest sequence in the batch
        max_length_com = 0
        if batch:
            for key in ['chosen', 'rejected']:
                current_max = max([len(item[key])+1 for item in batch])
                max_length_com = max(max_length_com, current_max)

        #process each item in the batch
        for item in batch:
            prompt = torch.tensor(item['prompt'])
            batch_data["prompt"].append(prompt)
            for key in ['chosen', 'rejected']:
                # Adjust padding to the max length of the batch
                sequence = item[key]
                padded = sequence + [pad_token_id] * (max_length_com - len(sequence))
                mask = torch.ones(len(padded)).bool()
                # Set mask for all padding tokens to False
                mask[len(sequence):] = False
                #set mask for the inputs tokens to False
                # +2 sets for the newline tokens before ###Response to false
                if mask_prompt_tokens:
                    mask[:prompt.shape[0]+2] = False
                batch_data[key].append(torch.tensor(padded))
                batch_data[f"{key}_mask"].append(mask)

        # Final processing of the batch data
        for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
            # Stake all sequences into a tnesor for given key
            tensor_stake  = torch.stack(batch_data[key])
            if allowed_max_length is not None:
                # Truncate to the allowed max length
                tensor_stake = tensor_stake[:, :allowed_max_length]
            batch_data[key] = tensor_stake.to(self.device)
        return batch_data

