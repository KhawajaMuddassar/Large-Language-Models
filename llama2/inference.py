import torch
from utils import generate_text, token_ids_to_text, text_to_token_ids, device
from llama2 import Llama2
from tokenizer import LlamaTokenizer
from utils import llama2_config

def main(llama2_config, input_prompt, device):
    device = device
    model = Llama2(llama2_config)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters Llama2: {total_params:,}")
    tokenizer = LlamaTokenizer()
    torch.manual_seed(123)
    token_ids = generate_text(
    model=model,
    idx=text_to_token_ids(input_prompt, tokenizer).to(device),
    max_new_tokens=30,
    context_size=llama2_config["context_length"],
    top_k=1,
    temperature=0.
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

if __name__ == '__main__':
    print(f'Warning: Ensure that at least 32GB of memory is available to load the Llama2 model.')
    input_prompt = 'it is good to be'
    main(llama2_config, input_prompt, device)
    