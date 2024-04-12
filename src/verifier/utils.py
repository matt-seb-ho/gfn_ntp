import json
import os

from dotenv import load_dotenv

from constants import DOTENV_PATH, HF_ACCESS_TOKEN_VAR_NAME, REPO_ROOT


def prepend_repo_root(p: str) -> str:
    return os.path.join(REPO_ROOT, p)

def _pp(d):
    print(json.dumps(d, indent=2))

def get_hf_access_token():
    load_dotenv(DOTENV_PATH)
    token = os.getenv(HF_ACCESS_TOKEN_VAR_NAME)
    return token

def add_pad_token(model, tokenizer, pad_token="<pad>", padding_side="right"):
    # CodeLlama/Llama2 does not have a default mask/pad token
    # https://github.com/TrelisResearch/llama-2-setup
    # Check if the pad token is already in the tokenizer vocabulary
    if '<pad>' not in tokenizer.get_vocab():
        # Add the pad token
        tokenizer.add_special_tokens({"pad_token": pad_token})

    # resize the embeddings
    model.resize_token_embeddings(len(tokenizer))

    # configure the pad token in the model
    model.config.pad_token_id = tokenizer.pad_token_id

    # check if they are equal
    assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"

    # print the pad token ids
    print('Tokenizer pad token ID:', tokenizer.pad_token_id)
    print('Model pad token ID:', model.config.pad_token_id)
    print('Model config pad token ID:', model.config.pad_token_id)
    
    # to prevent warnings
    tokenizer.padding_side = padding_side