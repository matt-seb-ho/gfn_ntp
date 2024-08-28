import torch
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler
import evaluate
from pathlib import Path
import random
from contextlib import contextmanager
from functools import partial
from typing import Optional
from collections import defaultdict
import numpy as np
import pandas as pd
from peft import PeftModel
from transformers import AutoTokenizer

import sys
print(sys.path)

from proof_flow.src.utils import prepare_environment_for_lean_dojo

from .proof_tree import extract_trajectories
from .replay_buffer import ReplayBuffer
from .reward import NTPReward
from .utils import base_to_lora, lora_to_base

prepare_environment_for_lean_dojo()
from lean_dojo import Dojo, TacticState, Theorem  # isort: skip

from .ntp import NeuralTheoremProvingTask
import torch
import wandb
from pathlib import Path


class NTP_PPO(NeuralTheoremProvingTask):
    def __init__(
        self,
        model: PeftModel,
        tokenizer: AutoTokenizer,
        reward: NTPReward,
        reward_buffer: ReplayBuffer,
        n_samples: int | list[int],
        lr: float,
        pf_temp_high: float,
        pf_temp_low: float,
        pf_temp_prob: float,
        use_buffer_prob: float,
        reward_temp_start: float,
        reward_temp_end: float,
        reward_temp_horizon: int,
        illegal_token_mask: np.ndarray,
        train_probes: Optional[list[Theorem]] = None,
        val_probes: Optional[list[Theorem]] = None,
        use_4bit: bool = False,
        max_tactics: int = 3,
        min_tactic_tokens: int = 2,
        max_tactic_tokens: int = 30,
        use_hf_generate: bool = True,
        save_dir: str = "ppo_model_checkpoints",
        wandb_log: bool = False,
        wandb_entity: str = "vincentzhu",
        wandb_project: str = "ntp-ppo-training",  
        mini_batch_size: int = 1,
        optimize_cuda_cache: bool = True,      
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            reward=reward,
            reward_buffer=reward_buffer,
            n_samples=n_samples,
            lr=lr,
            pf_temp_high=pf_temp_high,
            pf_temp_low=pf_temp_low,
            pf_temp_prob=pf_temp_prob,
            use_buffer_prob=use_buffer_prob,
            reward_temp_start=reward_temp_start,
            reward_temp_end=reward_temp_end,
            reward_temp_horizon=reward_temp_horizon,
            illegal_token_mask=illegal_token_mask,
            train_probes=train_probes,
            val_probes=val_probes,
            use_4bit=use_4bit,
            max_tactics=max_tactics,
            min_tactic_tokens=min_tactic_tokens,
            max_tactic_tokens=max_tactic_tokens,
            use_hf_generate=use_hf_generate,
        )

        # PPO-specific configuration
        self.ppo_config = PPOConfig(
            learning_rate=lr,
            batch_size=n_samples if isinstance(n_samples, int) else max(n_samples),
            mini_batch_size=mini_batch_size,  # You can adjust this as needed
            optimize_cuda_cache=optimize_cuda_cache,
        )

        # Initialize PPO Trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Create directory for saving model checkpoints
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb
        self.wandb_log = wandb_log
        if self.wandb_log:
            self.entity = wandb_entity
            self.project_name = wandb_project
            wandb.init(entity=self.entity, project=self.project_name, config=self.ppo_config)

    def training_step(self, theorem: Theorem, batch_idx):
        """
        Override the training step to use PPO for fine-tuning the model and save checkpoints.
        """
        # Step 1: Generate proof tree and collect log probabilities and rewards
        root, trajectories_logpf, log_r = self.forward(theorem)

        # Step 2: Extract the proof trajectories as text
        proof_texts = [trajectory["proof"] for trajectory in extract_trajectories(root, theorem.uid)]

        # Step 3: Perform PPO optimization step
        stats = self.ppo_trainer.step(proof_texts, log_r)

        # Step 4: Log PPO-specific metrics
        if self.wandb_log:
            wandb.log({
                "train/ppo_loss": stats["ppo/total_loss"],
                "train/ppo_policy_loss": stats["ppo/policy_loss"],
                "train/ppo_value_loss": stats["ppo/value_loss"],
                "train/ppo_entropy_loss": stats["ppo/entropy_loss"],
                "train/batch_idx": batch_idx,
            })

        # Step 5: Save model checkpoint
        if batch_idx % 100 == 0:  # Save every 100 batches, adjust as needed
            self.save_model(f"checkpoint_train_{batch_idx}")

        return stats["ppo/total_loss"]

    def validation_step(self, theorem: Theorem, batch_idx: int):
        """
        Override the validation step to use the PPO trainer and save the best model.
        """
        # Sample a proof and get the reward
        root, log_pf, log_r = self.forward(theorem)

        # Calculate loss using PPO's value and policy estimates
        loss, policy_loss, value_loss, entropy_loss = self.ppo_trainer.compute_ppo_loss(
            old_log_probs=log_pf,
            new_log_probs=self.ppo_trainer.policy_model(theorem),
            advantages=log_r - log_pf.sum(dim=-1),
            returns=log_r,
        )

        # Log metrics
        if self.wandb_log:
            wandb.log({
                "val/loss": loss,
                "val/logR": log_r.mean(),
                "val/ppo_policy_loss": policy_loss,
                "val/ppo_value_loss": value_loss,
                "val/ppo_entropy_loss": entropy_loss,
                "val/batch_idx": batch_idx,
            })

        # Save the best model based on validation loss
        if self.best_val_loss is None or loss < self.best_val_loss:
            self.best_val_loss = loss
            self.save_model("best_model")

        return loss

    def save_model(self, name: str):
        """
        Save the model parameters and log as wandb artifact.
        """
        save_path = self.save_dir / f"{name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.ppo_trainer.optimizer.state_dict(),
            'loss': self.best_val_loss,
        }, save_path)
        print(f"Model saved to {save_path}")

        # Log model as wandb artifact
        artifact = wandb.Artifact(name=f"model-{name}", type="model")
        artifact.add_file(str(save_path))
        wandb.log_artifact(artifact)

    def load_model(self, name: str):
        """
        Load the model parameters from a local file or wandb artifact.
        """
        load_path = self.save_dir / f"{name}.pt"
        if load_path.exists():
            checkpoint = torch.load(load_path)
        else:
            # Try to load from wandb artifact
            artifact = wandb.use_artifact(f"model-{name}:latest")
            artifact_dir = artifact.download()
            checkpoint = torch.load(Path(artifact_dir) / f"{name}.pt")

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ppo_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['loss']
        print(f"Model loaded: {name}")

    def on_train_end(self):
        """
        Finish the wandb run when training ends.
        """
        wandb.finish()
    




# Ignore: this was very general and not integrated into the reat of the code
class General_PPO_Fine_Tuner():
    """
    This is my first stab at the PPO implementation with trl PPO. The reward function is a loose integration
    with the BLEU score. This is a work in progress and will be refined as we go along.
    """
    def __init__(self, model_name, learning_rate=1e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure the tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # PPO config
        self.ppo_config = PPOConfig(
            learning_rate=learning_rate,
            batch_size=4,
            mini_batch_size=1,
            optimize_cuda_cache=True,
        )
        
        # PPO Trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        self.generation_kwargs = {
            "min_length": 32,
            "max_length": 128,
            "do_sample": True,
            "top_k": 0,
            "top_p": 0.92,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        # Load BLEU metric for evaluation
        # self.bleu = evaluate.load("bleu")

    def prepare_dataset(self, train_data, val_data, test_data):
        self.train_dataset = Dataset.from_dict(train_data)
        self.val_dataset = Dataset.from_dict(val_data)
        self.test_dataset = Dataset.from_dict(test_data)

    def reward_function(self, generated_code, reference_code):
        # Calculate BLEU score as a reward
        bleu_score = self.bleu.compute(predictions=[generated_code], references=[[reference_code]])
        return torch.tensor(bleu_score['bleu'])

    def train(self, num_epochs=1):
        for epoch in range(num_epochs):
            for batch in self.train_dataset.shuffle().iter(batch_size=self.ppo_config.batch_size):
                prompt_tensors = self.tokenizer(batch["prompt"], return_tensors="pt", padding=True).to(self.device)
                
                response_tensors = self.ppo_trainer.generate(
                    prompt_tensors.input_ids,
                    return_prompt=False,
                    **self.generation_kwargs,
                )
                                
                generated_code = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                rewards = [self.reward_function(gen, ref) for gen, ref in zip(generated_code, batch["code"])]
                rewards = torch.stack(rewards).to(self.device)
                
                self.ppo_trainer.step(prompt_tensors.input_ids, response_tensors, rewards)
            
            # Validation step
            self.validate()

    def validate(self):
        self.model.eval()
        total_bleu = 0
        with torch.no_grad():
            for batch in self.val_dataset:
                prompt = batch["prompt"]
                reference_code = batch["code"]
                
                generated_code = self.generate(prompt)
                
                bleu_score = self.bleu.compute(predictions=[generated_code], references=[[reference_code]])
                total_bleu += bleu_score['bleu']
        
        avg_bleu = total_bleu / len(self.val_dataset)
        print(f"Validation BLEU Score: {avg_bleu}")
        self.model.train()

    def test(self):
        self.model.eval()
        total_bleu = 0
        with torch.no_grad():
            for batch in self.test_dataset:
                prompt = batch["prompt"]
                reference_code = batch["code"]
                
                generated_code = self.generate(prompt)
                
                bleu_score = self.bleu.compute(predictions=[generated_code], references=[[reference_code]])
                total_bleu += bleu_score['bleu']
        
        avg_bleu = total_bleu / len(self.test_dataset)
        print(f"Test BLEU Score: {avg_bleu}")

    def save_model(self, path):
        # Save the model architecture and weights
        torch.save(self.model.state_dict(), path + ".pt")
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(path + "_tokenizer")
        
        print(f"Model saved to {path}.pt")
        print(f"Tokenizer saved to {path}_tokenizer")

    def load_model(self, path):
        # Load the model architecture and weights
        self.model.load_state_dict(torch.load(path + ".pt", map_location=self.device))
        self.model.to(self.device)
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path + "_tokenizer")
        
        print(f"Model loaded from {path}.pt")
        print(f"Tokenizer loaded from {path}_tokenizer")

    def generate(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids, **self.generation_kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)