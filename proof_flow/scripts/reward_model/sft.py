import argparse
import hydra
import json
import torch
import wandb
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
    prepare_environment_for_lean_dojo,
)
from proof_flow.src.search.proof_search import DistributedProver, Status
from proof_flow.src.prompts import PROMPT_DICT

prepare_environment_for_lean_dojo()
from lean_dojo import LeanGitRepo, Theorem


class SFTTrainerWithSearchEval(SFTTrainer):
    def __init__(self, *args, **kwargs):
        # additional parameters for proof search evaluation
        self.search_eval_params = kwargs.pop("search_eval_params")
        self.proof_search_eval_probes = kwargs.pop("proof_search_eval_probes")
        self.prompt_template = kwargs.pop("prompt_template")
        super().__init__(*args, **kwargs)
        

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        # setup prover object
        prover = DistributedProver(
            use_vllm=False, # use_vllm
            gen_ckpt_path="", # gen_ckpt_path (needs to be not None)
            ret_ckpt_path=None, # ret_ckpt_path
            indexed_corpus_path=None, # indexed_corpus_path
            max_inp_seq_len=self.search_eval_params.max_input_seq_len,
            max_oup_seq_len=self.search_eval_params.max_output_seq_len,
            length_penalty=self.search_eval_params.length_penalty,
            tactic=None, # tactic
            module=None, # module
            num_workers=self.search_eval_params.num_workers,
            num_gpus=self.search_eval_params.num_gpus,
            timeout=self.search_eval_params.timeout,
            max_expansions=self.search_eval_params.max_expansions,
            max_depth=self.search_eval_params.max_depth,
            num_sampled_tactics=self.search_eval_params.num_sampled_tactics,
            max_new_tokens=self.search_eval_params.max_new_tokens,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_template=self.prompt_template,
            is_decoder_only=True,
        )

        # get repo and theorems
        probes = self.proof_search_eval_probes
        repo = LeanGitRepo(probes[0]["url"], probes[0]["commit"])
        thms = [
            Theorem(repo, thm["file_path"], thm["full_name"]) 
            for thm in probes
        ]
        positions = [None] * len(thms) # ignored by HF tac gen

        # run proof search
        self.model.eval()
        with torch.no_grad():
            if self.args.bf16:
                with torch.cuda.amp.autocast("cuda", dtype=torch.bfloat16):
                    results = prover.search_unordered(repo, thms, positions)
            else:
                results = prover.search_unordered(repo, thms, positions)
        self.model.train()
        
        # process results, write to log, return metrics
        num_proved = 0
        for r in results:
            if r is not None and r.status == Status.PROVED:
                num_proved += 1
        return {"val/num_proved": num_proved}
        

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
    for i in range(len(batch["prompt"])):
        output_text.append(f"{batch['prompt'][i]}{batch['completion'][i]}")
    return output_text


def main():
    psr = argparse.ArgumentParser()
    psr.add_argument("--config", type=str, default="verifier_training")
    args = psr.parse_args()
    
    # load config
    config = get_config(config_name=args.config)
    # make output directory relative to repo root
    config_train_args = config.sft.model.training_args
    config_train_args.output_dir = repo_root() / config_train_args.output_dir

    # set up wandb
    wandb.init(project=config.sft.wandb_project)
    
    # load dataset
    dataset = load_from_disk(repo_root() / config.sft.data.formatted_dataset_dir)
    train_data = dataset["train"]
    if config.sft.data.same_train_val:
        val_data = dataset["train"]
    else:
        val_data = None

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
        torch_dtype=torch.float16,
        # torch_dtype="auto",
        quantization_config=bnb_config,
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

    with open(repo_root() / config.sft.search_eval.probe_file) as f:
        probes = json.load(f)
        probes = list(probes.values())
        
    search_eval_params = hydra.utils.instantiate(
        config.sft.search_eval.search_params
    )
    prompt_template = PROMPT_DICT[config.sft.data.prompt_dict_key]
    
    # trainer = SFTTrainer(
    trainer = SFTTrainerWithSearchEval(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        tokenizer=tokenizer,
        max_seq_length=config.sft.model.max_seq_length,
        packing=use_packing,
        formatting_func=formatting_func,
        search_eval_params=search_eval_params,
        proof_search_eval_probes=probes,
        prompt_template=prompt_template,
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
