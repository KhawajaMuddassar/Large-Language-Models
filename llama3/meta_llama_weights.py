import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))
import torch
from safetensors.torch import load_file
from llama2.tokenizer import hf_Login

class weights:
    def __init__(self,):
        self.login = hf_Login()        
    def combined_weights(self,model_type):
        comb_weights = {}
        for i in range(1,5):
            if model_type == 'Instruct':
                weights_file = self.login.download_tokenizer(                
                                repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                                filename=f"model-0000{i}-of-00004.safetensors",
                                local_dir="llama3/Llama-3-8B-Instruct") 
            elif model_type == 'Pretrained':
                weights_file = self.login.download_tokenizer(           
                                repo_id="meta-llama/Meta-Llama-3-8B",
                                filename=f"model-0000{i}-of-00004.safetensors",
                                local_dir="llama3/Llama-3-8B-Pretrained")
            else:
                print(f'Provide Model Type : "Instruct or Pretrained"')
            current_weights = load_file(weights_file)
            comb_weights.update(current_weights)
        return comb_weights
    
    def assign(self, left, right, tensor_name='unknown'):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        if isinstance(right, torch.Tensor):
            return torch.nn.Parameter(right.clone().detach())
        else:
            return torch.nn.Parameter(torch.tensor(right))
        
    def load_weights(self,model, param_config, params ):
        model.tok_emb.weight = self.assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
                      
        for l in range(param_config["n_layers"]):
            # Load attention weights
            model.trf_blocks[l].att.q_w.weight = self.assign(model.trf_blocks[l].att.q_w.weight,
                                                     params[f"model.layers.{l}.self_attn.q_proj.weight"],f"model.layers.{l}.self_attn.q_proj.weight")
            model.trf_blocks[l].att.k_w.weight = self.assign(model.trf_blocks[l].att.k_w.weight,
                                                    params[f"model.layers.{l}.self_attn.k_proj.weight"],f"model.layers.{l}.self_attn.k_proj.weight")
            model.trf_blocks[l].att.v_w.weight = self.assign(model.trf_blocks[l].att.v_w.weight,
                                                    params[f"model.layers.{l}.self_attn.v_proj.weight"],f"model.layers.{l}.self_attn.v_proj.weight")
            model.trf_blocks[l].att.out_proj.weight = self.assign(model.trf_blocks[l].att.out_proj.weight,
                                                    params[f"model.layers.{l}.self_attn.o_proj.weight"],f"model.layers.{l}.self_attn.o_proj.weight")
            model.trf_blocks[l].norm1.weights = self.assign(model.trf_blocks[l].norm1.weights,
                                                    params[f"model.layers.{l}.input_layernorm.weight"],f"model.layers.{l}.input_layernorm.weight")
            # Load FeedForward weights
            model.trf_blocks[l].ff.fc1.weight = self.assign(model.trf_blocks[l].ff.fc1.weight,
                                                    params[f"model.layers.{l}.mlp.gate_proj.weight"],f"model.layers.{l}.mlp.gate_proj.weight")
            model.trf_blocks[l].ff.fc2.weight = self.assign(model.trf_blocks[l].ff.fc2.weight,
                                                    params[f"model.layers.{l}.mlp.up_proj.weight"],f"model.layers.{l}.mlp.up_proj.weight")
            model.trf_blocks[l].ff.fc3.weight = self.assign(model.trf_blocks[l].ff.fc3.weight,
                                                    params[f"model.layers.{l}.mlp.down_proj.weight"],f"model.layers.{l}.mlp.down_proj.weight")
            model.trf_blocks[l].norm2.weights = self.assign(model.trf_blocks[l].norm2.weights,
                                                    params[f"model.layers.{l}.post_attention_layernorm.weight"],f"model.layers.{l}.post_attention_layernorm.weight")
        # Load output layer weights
        model.nl.weights = self.assign(model.nl.weights, params["model.norm.weight"], "model.norm.weight")
            
        if "lm_head.weight" in params.keys():
            model.output_head.weight = self.assign(model.output_head.weight, params["lm_head.weight"], "lm_head.weight")
        else:
            model.output_head.weight = self.assign(model.output_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
            print("Model uses weight tying.")


        
