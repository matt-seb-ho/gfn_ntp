import json
import os
import pickle
import hydra
import torch
from itertools import product
from datasets import Dataset
from huggingface_hub import login
from icecream import ic
from peft import LoraConfig
# from sft import _add_pad_token, sft_subset
import flash_attn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import DPOTrainer, DPOConfig

# from prep_sft_data import load_sft_data, sft_subset
from proof_flow.src.utils import get_hf_access_token, repo_root, get_config, set_up_padding
from proof_flow.src.prompts import INSTRUCTION_PROMPT_TEMPLATE, DEEPSEEK_RM_ST_PROMPT_TEMPLATE, RM_TEMPLATES

# constants
# DPO_DATA_PATH = repo_root() / "data/paired_random_train.json"
# DPO_BETA = 0.01 # TODO: tune this
# USE_SUBSET = True
# DPO_SUBSET_PATH = None
# DPO_SUBSET_EXISTS = False
# DPO_SUBSET_SIZE = 10000
# BASE_MODEL_ID = "EleutherAI/llemma_7b"
# HF_ACCESS_TOKEN = get_hf_access_token("src/.env", True)

HF_ACCESS_TOKEN = get_hf_access_token()
SANITY_CHECK = False


def expand_record(record):
    expanded_records = []
    positive_list = record.get("positive", [])
    negative_list = record.get("negative", [])
    
    # Generate the Cartesian product of positive and negative lists
    for pos, neg in product(positive_list, negative_list):
        # Create a new record for each combination
        new_record = record.copy()
        new_record["positive"] = pos['tactic']
        new_record["negative"] = neg['tactic']
        expanded_records.append(new_record)
    
    return expanded_records   

def flatten_record(record):
    flattened_records = []
    positive_list = record.get("positive", [])
    negative_list = record.get("negative", [])
    
    # Check if there are elements in both lists
    if positive_list and negative_list:
        # Get only the first combination of positive and negative tactics
        pos = positive_list[0]
        neg = negative_list[0]
        # Create a new record for the first combination
        new_record = record.copy()
        new_record["positive"] = pos['tactic']
        new_record["negative"] = neg['tactic']
        flattened_records.append(new_record)
    
    return flattened_records 


def load_dpo_data(cfg):
    # load raw data into arrow format
    with open(repo_root() / cfg.dpo.data.formatted_dataset_dir) as f:
        records = json.load(f)
    
    processed_records = []
    for record in records:
        if cfg.dpo.data.expand_records:
            processed_records += expand_record(record)
        else:
            processed_records += flatten_record(record)
    
    full_dataset = Dataset.from_list(processed_records)

    # prep for DPOTrainer format
    def add_prompt_template(example):
        # example["prompt"] = INSTRUCTION_PROMPT_TEMPLATE.format(state=example["state"])
        # example["prompt"] = DEEPSEEK_RM_ST_PROMPT_TEMPLATE.format(state=example["state"])
        st_or_sts = "sts" if cfg.dpo.data.use_sts_format else "st"
        example["prompt"] = RM_TEMPLATES[cfg.dpo.data.model_prompt_template][st_or_sts]["prompt"].format(state=example["state_before"])
        return example
    
    dataset = (
        full_dataset.map(add_prompt_template)
            .rename_column("positive", "chosen")
            .rename_column("negative", "rejected")
    )
    # if USE_SUBSET:
    #     dataset = sft_subset(dataset, size=DPO_SUBSET_SIZE, file=DPO_SUBSET_PATH)
    
    return dataset


def main():
    # load config
    cfg = get_config(config_name="dpo_training")
    # make output directory relative to repo root
    config_train_args = cfg.dpo.model.training_args
    config_train_args.output_dir = repo_root() / config_train_args.output_dir
    
    # load dataset
    if os.path.exists(cfg.dpo.data.processed_data):
        with open(cfg.dpo.data.processed_data, "rb") as pkl:
            dataset = pickle.load(pkl)
    else:
        dataset = load_dpo_data(cfg)
        with open(cfg.dpo.data.processed_data, "wb") as pkl:
            pickle.dump(dataset, pkl)
        
    # test data loading
    if SANITY_CHECK:
        ic(dataset.column_names, len(dataset))
        ex0 = dataset[0]
        print("Printing example prompt/chosen/rejected:")
        print(ex0["prompt"])
        print(ex0["chosen"])
        print(ex0["rejected"])
        return

    # -- set up configs --
    # # int-4 config
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True, 
    #     bnb_4bit_use_double_quant=True, 
    #     bnb_4bit_quant_type="nf4", 
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    # # hyperparameters from LoRA/QLoRA papers
    # peft_config = LoraConfig(
    #     lora_alpha=128,
    #     lora_dropout=0.05,
    #     r=256,
    #     bias="none",
    #     target_modules="all-linear",
    #     task_type="CAUSAL_LM",
    # )
    
    # int-4 config
    bnb_config = hydra.utils.instantiate(cfg.dpo.model.bnb)
    # hyperparameters from LoRA/QLoRA papers
    peft_config = hydra.utils.instantiate(cfg.dpo.model.lora)


    # need to log in to hf acct to push checkpoints to hub
    login(token=HF_ACCESS_TOKEN)
    training_args = DPOConfig(
        output_dir=config_train_args.output_dir,                                        # directory to save and repository id
        num_train_epochs=config_train_args.num_train_epochs,                            # number of training epochs
        per_device_train_batch_size=config_train_args.per_device_train_batch_size,      # batch size per device during training
        gradient_accumulation_steps=config_train_args.gradient_accumulation_steps,      # number of steps before performing a backward/update pass
        gradient_checkpointing=config_train_args.gradient_checkpointing,                # use gradient checkpointing to save memory
        optim=config_train_args.optim,                                                  # use fused adamw optimizer
        logging_steps=config_train_args.logging_steps,                                  # log every 10 steps
        save_strategy=config_train_args.save_strategy,                                  # save checkpoint every epoch
        learning_rate=config_train_args.learning_rate,                                  # learning rate, based on QLoRA paper
        bf16=config_train_args.bf16,                                                    # use bfloat16 precision
        tf32=config_train_args.tf32,                                                    # use tf32 precision
        max_grad_norm=config_train_args.max_grad_norm,                                  # max gradient norm based on QLoRA paper
        warmup_ratio=config_train_args.warmup_ratio,                                    # warmup ratio based on QLoRA paper
        lr_scheduler_type=config_train_args.lr_scheduler_type,                          # use constant learning rate scheduler
        push_to_hub=config_train_args.push_to_hub,                                      # push model to hub
        report_to=config_train_args.report_to, 
        beta=config_train_args.beta,
    )

    # -- load model and tokenizer --
    model = AutoModelForCausalLM.from_pretrained(
        cfg.dpo.model.base_model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    # with peft, ref model is optional for dpo
    # model_ref = AutoModelForCausalLM.from_pretrained(
    #     BASE_MODEL_ID,
    #     device_map="auto",
    #     attn_implementation="flash_attention_2",
    #     torch_dtype=torch.bfloat16,
    #     quantization_config=bnb_config,
    # )
    tokenizer = AutoTokenizer.from_pretrained(cfg.dpo.model.base_model_id)
    # _add_pad_token(model, tokenizer)
    set_up_padding(model, tokenizer)

    # -- initialize trainer --
    dpo_trainer = DPOTrainer(
        model=model,
        # ref_model=model_ref,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        # packing=cfg.dpo.model.packing, # not an actual arugment in DPOTrainer
        # dataset_kwargs={
        #     "add_special_tokens": False,  # We template with special tokens
        #     "append_concat_token": False, # No need to add additional separator token
        # }
    )
    
    # -- train, save, and clean up --
    dpo_trainer.train()
    dpo_trainer.save_model()
    dpo_trainer.push_to_hub()
    
    del model
    del dpo_trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
