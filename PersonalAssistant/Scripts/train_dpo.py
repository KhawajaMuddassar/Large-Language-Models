import torch
import json
import tiktoken
import time
from pathlib import Path
from process_data import ProcessDataDPO
from gpt_model import GPTModel
from utils import (dpo_batch_loss,
                   evaluate_dpo_loader_loss,
                   generate_and_print_sample,
                   dpo_plot_losses,
                   format_input_to_alpaca)

class train_dpo:
    def __init__(self,):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.instruction_data = 'PersonalAssistant/data/processed/ollama_data_dedup.json'
        self.finetuned_model_path = 'PersonalAssistant/model/modelfiles/FT_Model.pth'
        self.json_config = 'PersonalAssistant/model_config.json'
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.num_epochs = 1
        self.policy_model_path = 'PersonalAssistant/model/modelfiles/dpo_Model.pth'
        self.policy_model, self.reference_model = self.models()
        with open(self.instruction_data, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} instructions from {self.json_config}")
        self.train_loader, self.val_loader, self.val, self.test = ProcessDataDPO(data).GetDataReady()

    def models(self,):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(self.json_config, 'r') as file:
            config = json.load(file)
        choose_model = 'gpt2-medium'
        if choose_model:
            print(f'Loading {choose_model} as policy model...\n')
            config['base_config'].update(config['model_config'][choose_model])
            policy_model = GPTModel(config['base_config'])
            policy_model.load_state_dict(torch.load(self.finetuned_model_path,
                                 weights_only=True,
                                 map_location=device))
            policy_model.eval()

            print(f'Loading {choose_model} as reference model...\n')
            reference_model = GPTModel(config['base_config'])
            reference_model.load_state_dict(torch.load(self.finetuned_model_path,
                                      weights_only=True,
                                      map_location=device))
            reference_model.eval()
            return policy_model.to(device), reference_model.to(device)

    def training_loop(self, start_context, beta=0.1, eval_freq=25, eval_iter=10 ):
        optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=5e-6, weight_decay=0.01)       
        tracking = {
            "train_losses": [],
            "train_chosen_rewards": [],
            "train_rejected_rewards": [],
            "val_losses": [],
            "val_chosen_rewards": [],
            "val_rejected_rewards": [],
            "tokens_seen": []
            }
        tokens_seen, global_step = 0, -1

        # Main training loop
        for epoch in range(self.num_epochs):
            self.policy_model.train()  

            for batch_idx, batch in enumerate(self.train_loader):
                optimizer.zero_grad() 
                loss, chosen_rewards, rejected_rewards = dpo_batch_loss(
                batch=batch,
                policy_model=self.policy_model,
                reference_model=self.reference_model,
                beta=beta
                )
                loss.backward()  
                optimizer.step() 
                tokens_seen += batch["chosen"].numel()
                global_step += 1
                # Evaluation steps
                if global_step % eval_freq == 0:
                    res = evaluate_dpo_loader_loss(
                    policy_model=self.policy_model,
                    reference_model=self.reference_model,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    beta=beta,
                    eval_iter=eval_iter
                    )
                    tracking["train_losses"].append(res["train_loss"])
                    tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                    tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                    tracking["val_losses"].append(res["val_loss"])
                    tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                    tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                    tracking["tokens_seen"].append(tokens_seen)
                    train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                    val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]
                    print(
                        f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                        f"Train reward margins {train_reward_margin:.3f}, "
                        f"Val reward margins {val_reward_margin:.3f}"
                        )

            # Print a sample text after each epoch
            generate_and_print_sample(
                    model=self.policy_model,
                    tokenizer=self.tokenizer,
                    device=loss.device,
                    start_context=start_context
                )
        return tracking

    def train(self,):
        start_time = time.time()
        evaluate_dpo = evaluate_dpo_loader_loss(
                                policy_model=self.policy_model,
                                reference_model=self.reference_model,
                                train_loader=self.train_loader,
                                val_loader=self.val_loader,
                                beta=0.1,
                                eval_iter=5
                                )
        print("\nTraining loss:", evaluate_dpo["train_loss"])
        print("Validation loss:", evaluate_dpo["val_loss"])
        print("Train reward margin:", evaluate_dpo["train_chosen_reward"] - evaluate_dpo["train_rejected_reward"])
        print("Val reward margin:", evaluate_dpo["val_chosen_reward"] - evaluate_dpo["val_rejected_reward"],'\n')

        tracking = self.training_loop(start_context=format_input_to_alpaca(self.val[2]))

         # Save model
        torch.save(self.policy_model.state_dict(), self.policy_model_path)
        print(f"DPO Model saved ")

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")

        fig_loc = 'dpo_loss.pdf'
        epochs_tensor = torch.linspace(0, self.num_epochs, len(tracking["train_losses"]))
        dpo_plot_losses(
                epochs_seen=epochs_tensor,
                tokens_seen=tracking["tokens_seen"],
                train_losses=tracking["train_losses"],
                val_losses=tracking["val_losses"],
                fig_loc = fig_loc
                )
        fig_loc = 'margin.pdf'
        train_reward_margins = [i-j for i,j in zip(tracking["train_chosen_rewards"], tracking["train_rejected_rewards"])]
        val_reward_margins = [i-j for i,j in zip(tracking["val_chosen_rewards"], tracking["val_rejected_rewards"])]
        dpo_plot_losses(
                epochs_seen=epochs_tensor,
                tokens_seen=tracking["tokens_seen"],
                train_losses=train_reward_margins,
                val_losses=val_reward_margins,
                fig_loc = fig_loc
                )

train_dpo().train()