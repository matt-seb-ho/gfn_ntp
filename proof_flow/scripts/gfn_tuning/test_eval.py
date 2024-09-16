import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from types import MethodType
from proof_flow.scripts.gfn_tuning.train import (
    get_model, 
    get_reward,
    get_val_probes,
)
from proof_flow.src.prompts import (
    DEEPSEEK_RM_ST_PROMPT_TEMPLATE_V2,
    DEEPSEEK_RM_ST_PROMPT_TEMPLATE_V3,
)
from proof_flow.src.gfn_tuning.reward import NTPReward
from proof_flow.src.gfn_tuning.replay_buffer import ReplayBuffer
from proof_flow.src.gfn_tuning.lean_data_module import NTPDataModule
from proof_flow.src.gfn_tuning.ntp import NeuralTheoremProvingTask
from proof_flow.src.utils import (
    get_config,
    disable_tokenizer_parallelism,
    set_up_debug_logging,
)
from proof_flow.src.search.proof_search import Status
from loguru import logger


# relative to this file (proof_flow/scripts/gfn_tuning/train.py)
CONFIG_DIR = "../../../configs/"


def main(config: DictConfig):
    debug_log_level = set_up_debug_logging(config.task.debug_logger)
    disable_tokenizer_parallelism()
    pl.seed_everything(config.seed, workers=True)

    model, tokenizer = get_model(config)
    # if config.task.training.use_4bit:
    #     # bnb_config = BitsAndBytesConfig(
    #     #     load_in_4bit=True,
    #     #     bnb_4bit_quant_type="nf4",
    #     #     bnb_4bit_compute_dtype="float16",
    #     #     bnb_4bit_use_double_quant=True,
    #     # )
    #     # config has all the same options as above
    #     # EXCEPT bnb_4bit_compute_dtype is "bfloat16" instead of "float16"
    #     bnb_config = hydra.utils.instantiate(config.task.model.bnb)
    # else:
    #     bnb_config = None

    # tokenizer = AutoTokenizer.from_pretrained(config.task.model.name)
    # model = AutoModelForCausalLM.from_pretrained(
    #     config.task.model.name, 
    #     torch_dtype="auto", # defer to torch_dtype from model config.json
    #     device_map="auto", 
    #     quantization_config=bnb_config,
    # )
    
    reward = get_reward(config, model, tokenizer)
    reward_buffer = ReplayBuffer(
        buffer_size=config.task.reward.buffer_size,
        termination_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        sim_tolerance=config.task.reward.buffer_sim_tolerance,
    )
    data = NTPDataModule(
        data_path=config.task.data.path,
        train_size=config.task.data.train_size,
    )
    data.setup("fit")
    val_probes = get_val_probes(config)

    task = NeuralTheoremProvingTask(
        model=model,
        tokenizer=tokenizer,
        reward=reward,
        reward_buffer=reward_buffer,
        n_samples=config.task.training.n_samples,
        lr=config.task.training.lr,
        pf_temp_high=config.task.training.pf_temp_high,
        pf_temp_low=config.task.training.pf_temp_low,
        pf_temp_prob=config.task.training.pf_temp_prob,
        use_buffer_prob=config.task.training.use_buffer_prob,
        reward_temp_start=config.task.reward.temp_start,
        reward_temp_end=config.task.reward.temp_end,
        reward_temp_horizon=config.task.reward.temp_horizon,
        use_4bit=config.task.training.use_4bit,
        max_tactics=config.task.constraints.max_tactics,
        min_tactic_tokens=config.task.constraints.min_tactic_tokens,
        max_tactic_tokens=config.task.constraints.max_tactic_tokens,
        model_inference_batch_size=config.task.model.inf_batch_size,
        branch_only_at_root=config.task.training.branch_only_at_root,
        dojo_timeout=config.task.training.dojo_timeout,
        search_eval_probes=val_probes,
        search_eval_cfg=hydra.utils.instantiate(config.task.search_eval_cfg),
        ckpt_dest=config.task.training.ckpt_dest,
        save_ckpt_on_val=config.task.training.save_ckpt_on_val,
        sanity_check_probes=config.task.search_eval.sanity_check_probe_count,
        debug_log_level=debug_log_level,
        tac_gen_prompt_template=DEEPSEEK_RM_ST_PROMPT_TEMPLATE_V2,
    )

    trainer = pl.Trainer(
        accelerator=config.device.accelerator,
        max_epochs=config.task.training.epochs,
        accumulate_grad_batches=config.task.training.accumulate_grad_batches,
        logger=config.logger
        if isinstance(config.logger, bool)
        else hydra.utils.instantiate(config.logger),
        callbacks=[hydra.utils.instantiate(c) for c in config.task.callbacks],
    )

    # Fix a bug that arises when using 4-bit quantized models.
    # It's caused by different operations being on different devices,
    # so we'll just deactivate lightning's automatic device placement
    # and let huggingface handle the dynamic device placement
    if config.task.training.use_4bit:
        task.to = MethodType(lambda s, _: s, task)
        task.cuda = MethodType(lambda s: s, task)

    # trainer.fit(model=task, datamodule=data)
    model.eval()
    results = task.run_proof_search_eval()
    name_to_idx = {
        thm_dict["full_name"]: thm_idx
        for thm_idx, thm_dict in enumerate(val_probes)
    }
    for r in results:
        if r is None:
            print("Discarded")
            continue
        # r is a SearchResult object
        proved = (r.status == Status.PROVED)
        thm_dict = val_probes[name_to_idx[r.theorem.full_name]]
        proof_len = len(thm_dict["traced_tactics"])
        logger.info(f"proved: {proved}, thm: {thm_dict['full_name']}, proof_len: {proof_len}")
        

if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
