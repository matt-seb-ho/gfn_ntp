import hydra
import torch
from datasets import load_from_disk
from huggingface_hub import login
from omegaconf import OmegaConf
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer

from proof_flow.src.utils import (
    set_up_padding, 
    get_config, 
    get_hf_access_token,
    repo_root,
)

# code based on:
# https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
# (from the blog of HF's tech lead)

# prompt formatting references:
# https://huggingface.co/docs/trl/en/sft_trainer#format-your-input-prompts
# https://github.com/huggingface/trl/pull/444#issue-1760952763
# key points
# - packed datset requires non-batched formatting_func
# - non-packed dataset requires batched formatting_func
def sft_formatting_func(example):
    return f"{example['prompt']}{example['completion']}"


def batch_sft_formatting_func(batch):
    output_text = []
    for example in batch:
        output_text.append(f"{example['prompt']}{example['completion']}")
    return output_text


def main():
    # load config
    config = get_config(config_name="verifier_training")
    # make output directory relative to repo root
    config_train_args = config.sft.model.training_args
    config_train_args.output_dir = repo_root() / config_train_args.output_dir
    
    # load dataset
    dataset = load_from_disk(repo_root() / config.sft.data.formatted_dataset_dir)
    train_data = dataset["train"]

    # Hugging Face model id
    model_id = config.sft.model.base_model_id
    
    # BitsAndBytesConfig int-4 config
    # bnb_file_cfg = OmegaConf.to_container(config.sft.model.bnb)
    # bnb_file_cfg["bnb_4bit_compute_dtype"] = getattr(torch, bnb_file_cfg["bnb_4bit_compute_dtype"])
    # bnb_config = BitsAndBytesConfig(**bnb_file_cfg)
    bnb_config = hydra.utils.instantiate(config.sft.model.bnb)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    set_up_padding(model, tokenizer)

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    # peft_config = LoraConfig(**config.sft.model.lora)
    peft_config = hydra.utils.instantiate(config.sft.model.lora)

    training_args = TrainingArguments(
        **config.sft.model.training_args
    )

    # login to the hub (needed for saving the model)
    login(token=get_hf_access_token())

    use_packing = config.sft.model.packing
    print(f"Using packed dataset: {use_packing}")
    formatting_func = sft_formatting_func if use_packing else batch_sft_formatting_func
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        peft_config=peft_config,
        tokenizer=tokenizer,
        max_seq_length=config.sft.model.max_seq_length,
        packing=use_packing,
        formatting_func=formatting_func,
    )

    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()
    
    # save model
    trainer.save_model()

    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()
    
            
if __name__ == "__main__":
    main()
