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
        model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
        assign = assign()        
        for l in range(param_config["n_layers"]):
            # Load attention weights
            model.trf_blocks[l].att.W_query.weight = assign(model.trf_blocks[l].att.W_query.weight,
                                                     params[f"model.layers.{l}.self_attn.q_proj.weight"],f"model.layers.{l}.self_attn.q_proj.weight")
            model.trf_blocks[l].att.W_key.weight = assign(model.trf_blocks[l].att.W_key.weight,
                                                    params[f"model.layers.{l}.self_attn.k_proj.weight"],f"model.layers.{l}.self_attn.k_proj.weight")
            model.trf_blocks[l].att.W_value.weight = assign(model.trf_blocks[l].att.W_value.weight,
                                                    params[f"model.layers.{l}.self_attn.v_proj.weight"],f"model.layers.{l}.self_attn.v_proj.weight")
            model.trf_blocks[l].att.out_proj.weight = assign(model.trf_blocks[l].att.out_proj.weight,
                                                    params[f"model.layers.{l}.self_attn.o_proj.weight"],f"model.layers.{l}.self_attn.o_proj.weight")
            model.trf_blocks[l].norm1.weight = assign(model.trf_blocks[l].norm1.weight,
                                                    params[f"model.layers.{l}.input_layernorm.weight"],f"model.layers.{l}.input_layernorm.weight")
            # Load FeedForward weights
            model.trf_blocks[l].ff.fc1.weight = assign(model.trf_blocks[l].ff.fc1.weight,
                                                    params[f"model.layers.{l}.mlp.gate_proj.weight"],f"model.layers.{l}.mlp.gate_proj.weight")
            model.trf_blocks[l].ff.fc2.weight = assign(model.trf_blocks[l].ff.fc2.weight,
                                                    params[f"model.layers.{l}.mlp.up_proj.weight"],f"model.layers.{l}.mlp.up_proj.weight")
            model.trf_blocks[l].ff.fc3.weight = assign(model.trf_blocks[l].ff.fc3.weight,
                                                    params[f"model.layers.{l}.mlp.down_proj.weight"],f"model.layers.{l}.mlp.down_proj.weight")
            model.trf_blocks[l].norm2.weight = assign(model.trf_blocks[l].norm2.weight,
                                                    params[f"model.layers.{l}.post_attention_layernorm.weight"],f"model.layers.{l}.post_attention_layernorm.weight")
        # Load output layer weights
        model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")
            
        if "lm_head.weight" in params.keys():
            model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
        else:
            model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
            print("Model uses weight tying.")

#model = llama3(llama3_config_8B)
wgt = weights()
comb_wgt = wgt.combined_weights(model_type='Instruct')
print(list(comb_wgt.keys())[:15])
#load_wgt = wgt.load_weights(model,llama3_config_8B, comb_wgt)
#model.to(device);
#del comb_wgt  # free up memory
        
