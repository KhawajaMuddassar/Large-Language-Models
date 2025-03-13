from huggingface_hub import login, hf_hub_download
import json
import os
import sentencepiece as spm

class hf_Login:
    def __init__(self, config_path="llama2/config.json"):
        self.config_path = config_path
        self.access_token = None

    def _load_access_token(self):
        """Load the access token from the config file."""
        try:
            with open(self.config_path, "r") as config_file:
                config = json.load(config_file)
                self.access_token = config.get("HF_ACCESS_TOKEN")
                if not self.access_token:
                    raise ValueError("Access token not found in the config.json file.")
        except FileNotFoundError:
            raise FileNotFoundError(f"{self.config_path} not found.")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in the config file.")

    def login_to_huggingface(self):
        """Login to HuggingFace using the access token."""
        if self.access_token is None:
            self._load_access_token()
        login(token=self.access_token)

    def download_tokenizer(self, repo_id="meta-llama/Llama-2-7b", filename="tokenizer.model", local_dir="llama2/Llama-2-7b"):
        os.makedirs(local_dir, exist_ok=True)
        """Download the tokenizer file from HuggingFace."""
        tokenizer_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir
        )
        return tokenizer_file

class LlamaTokenizer:
    def __init__(self, config_path="llama2/config.json", repo_id="meta-llama/Llama-2-7b", filename="tokenizer.model", local_dir="llama2/Llama-2-7b"):        
        self.login_instance = hf_Login(config_path)
        self.login_instance.login_to_huggingface()        
        self.tokenizer_file = self.login_instance.download_tokenizer(repo_id, filename, local_dir)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.tokenizer_file)
    def encode(self, text):
        return self.sp.encode_as_ids(text)
    def decode(self, ids):
        return self.sp.decode_pieces(ids)   
