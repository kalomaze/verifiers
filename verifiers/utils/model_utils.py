from importlib.util import find_spec
from typing import Dict, Any, Union, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def is_liger_available() -> bool:
    return find_spec("liger_kernel") is not None

def get_model(model_name: str, model_kwargs: Union[Dict[str, Any], None] = None) -> Any:
    if model_kwargs is None:
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
    if is_liger_available():
        print("Using Liger kernel")
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        return AutoLigerKernelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

def get_tokenizer(model_name: str, use_chat_template: bool = True) -> Any:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(f"{model_name}-Instruct")
        except Exception:
            raise ValueError(f"Failed to load tokenizer for {model_name}")

    # Only override if explicitly disabling chat template
    if not use_chat_template:
        tokenizer.chat_template = "{{ messages[0]['content'] }}"  # Use first message as raw text

    return tokenizer

def get_model_and_tokenizer(
    model_name: str,
    model_kwargs: Union[Dict[str, Any], None] = None,
    use_chat_template: bool = True
) -> Tuple[Any, Any]:
    model = get_model(model_name, model_kwargs)
    tokenizer = get_tokenizer(model_name, use_chat_template)
    return model, tokenizer
