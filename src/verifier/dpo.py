import json

import torch
from datasets import Dataset
from huggingface_hub import login
from icecream import ic
from peft import LoraConfig
from prompts import INSTRUCTION_PROMPT_TEMPLATE
from sft import _add_pad_token, sft_subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import DPOTrainer
# from prep_sft_data import load_sft_data, sft_subset
from src.utils import get_hf_access_token, make_path_relative_to_repo

# constants
DPO_DATA_PATH = make_path_relative_to_repo("data/paired_random_train.json")
DPO_BETA = 0.01 # TODO: tune this
USE_SUBSET = True
DPO_SUBSET_PATH = None
DPO_SUBSET_EXISTS = False
DPO_SUBSET_SIZE = 10000
BASE_MODEL_ID = "EleutherAI/llemma_7b"
HF_ACCESS_TOKEN = get_hf_access_token("src/.env", True)
# testing phase
SANITY_CHECK = False


def load_dpo_data():
    # load raw data into arrow format
    with open(DPO_DATA_PATH) as f:
        records = json.load(f)
    full_dataset = Dataset.from_list(records["pairs"])

    # prep for DPOTrainer format
    def add_prompt_template(example):
        example["prompt"] = INSTRUCTION_PROMPT_TEMPLATE.format(state=example["state"])
        return example
    dataset = (
        full_dataset.map(add_prompt_template)
            .rename_column("positive", "chosen")
            .rename_column("negative", "rejected")
    )
    if USE_SUBSET:
        dataset = sft_subset(dataset, size=DPO_SUBSET_SIZE, file=DPO_SUBSET_PATH)
    
    return dataset


def main():
    # test data loading
    dataset = load_dpo_data()
    if SANITY_CHECK:
        ic(dataset.column_names, len(dataset))
        ex0 = dataset[0]
        print("Printing example prompt/chosen/rejected:")
        print(ex0["prompt"])
        print(ex0["chosen"])
        print(ex0["rejected"])
        return

    # -- set up configs --
    # int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # hyperparameters from LoRA/QLoRA papers
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # need to log in to hf acct to push checkpoints to hub
    login(token=HF_ACCESS_TOKEN)
    args = TrainingArguments(
        output_dir="llemma_dpo_output",         # directory to save and repository id
        num_train_epochs=1,                     # number of training epochs
        per_device_train_batch_size=3,          # batch size per device during training
        gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-4,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=True,                       # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
    )

    # -- load model and tokenizer --
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
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
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    _add_pad_token(model, tokenizer)

    # -- initialize trainer --
    dpo_trainer = DPOTrainer(
        model=model,
        # ref_model=model_ref,
        ref_model=None,
        args=args,
        beta=DPO_BETA,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        # packing=True,
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
