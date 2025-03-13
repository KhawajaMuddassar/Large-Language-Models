import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))
import tiktoken
from pathlib import Path
from tiktoken.load import load_tiktoken_bpe
from typing import (
    Literal,
    Sequence,
    TypedDict
    )

class Tokenizer:
    def __init__(self, model_path):
        assert os.path.isfile(model_path), f'Model file {model_path} not found'
        merageable_ranks = load_tiktoken_bpe(model_path)
        self.special_tokens = {
            '<|begin_of_text|>':128000,
            '<|end_of_text|>': 128001,
            '<|start_header_id|>': 128006,
            '<|end_header_id|>': 128007,
            '<|eot_id|>': 128009,
        }
        self.special_tokens.update({
              128002 + i for i in range(256) if (128002 + i) not in self.special_tokens.values()
        })
        self.model = tiktoken.Encoding(
            name= Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=merageable_ranks,
            special_tokens= self.special_tokens
        )
    def encode(self, text, bos=False, eos=False, allowed_special=set(), disallowed_special=()):
        if bos:
            tokens = [self.special_tokens['<|begin_of_text|>']]
        else:
            tokens = []
        tokens += self.encode(text=text,allowed_special=allowed_special, disallowed_special=disallowed_special)
        if eos:
            tokens.append(self.special_tokens['<|end_of_text|>'])
        return tokens
    def decode(self, tokens):
        return self.model.decode(tokens)


Role = Literal['user', 'system','assistant']

class Message(TypedDict):
    role: Role
    content:str

Dialog = Sequence[Message]

class ChatFormat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def encode_header(self, message:Message):
        tokens = []
        tokens.append(self.tokenizer.special_tokens['<|start_header_id|>'])
        tokens.extend(self.tokenizer.encode(message['role'], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens['<|end_header_id|>'])
        tokens.extend(self.tokenizer.encode('\n\n',bos=False, eos=False))
        return tokens
    def encode_message(self, message:Message):
        tokens = self.encode_header(message)
        tokens.extend(self.tokenizer.encode(message['content'].strip(), bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens['<|eot_id|>'])
        return tokens
    def encode_dialog_prompt(self, diaglog:Dialog):
        tokens = []
        tokens.append(self.tokenizer.special_tokens['<|begin_of_text|>'])
        for message in diaglog:
            tokens.extend(self.encode_message(message))
        # start os assistant message for the model to complete
        tokens.extend(self.encode_header({'role':'assistant','content':''}))
        return tokens
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
    