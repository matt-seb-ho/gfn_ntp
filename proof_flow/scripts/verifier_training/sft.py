import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

# from prep_sft_data import load_sft_data, sft_subset
from proof_flow.src.utils import add_pad_token, get_hf_access_token, repo_root

# constants
SFT_DATA_PATH = repo_root() / "data/sfttif_random_train.json"
USE_SUBSET = True
SFT_SUBSET_PATH = None
SFT_SUBSET_EXISTS = False
SFT_SUBSET_SIZE = 10000
BASE_MODEL_ID = "EleutherAI/llemma_7b"
HF_ACCESS_TOKEN = get_hf_access_token("src/.env", True)


"""
# -- completion only example --
dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)
trainer.train()
"""

def _load_data_wrapper():
    if USE_SUBSET:
        if SFT_SUBSET_EXISTS:
            subset = load_sft_data(SFT_DATA_PATH)
            return subset
        else:
            full_dataset = load_sft_data(SFT_DATA_PATH)
            subset = sft_subset(full_dataset, size=SFT_SUBSET_SIZE, file=SFT_SUBSET_PATH)
            return subset
    else:
        dataset = load_sft_data(SFT_DATA_PATH)
        return dataset



def main():
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
        output_dir="llemma_sft_output",         # directory to save and repository id
        num_train_epochs=3,                     # number of training epochs
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

    # -- load model, tokenizer, and data --
    dataset = _load_data_wrapper()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    add_pad_token(model, tokenizer)

    # -- initialize trainer --
    trainer = SFTTrainer(
        model=model,
        args=args,
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
    trainer.train()
    trainer.save_model()
    
    del model
    del trainer
    torch.cuda.empty_cache()


# migrated/duplicated from data prep script 
# so I don't have to install leandojo dependencies when training on cloud
def load_sft_data(path):
    dataset = load_dataset("json", data_files=path, split="train")
    # -- sanity check --
    # ic(len(dataset))
    # ic(dataset[0])
    return dataset


def sft_subset(dataset, size, seed=42, file=None):
    subset = dataset.train_test_split(test_size=size, seed=seed)["test"]
    if file:
        subset.to_json(file, orient="records")
    return subset


if __name__ == "__main__":
    main()
