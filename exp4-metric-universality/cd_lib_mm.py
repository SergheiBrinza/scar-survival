#!/usr/bin/env python3
"""Multi-model wrapper around the cd_lib core from experiment 1.

Supports different families (Qwen, Phi, ...): apply_chat_template + extended set of stop tokens.

The original cd_lib lives in the experiment-1 folder (basic scar-survival scenario).
Set the EXP1_DIR environment variable to its location, or edit the EXP1_DIR constant
below.
"""
import os
import sys
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# Path to the experiment-1 folder that contains cd_lib.py, facts.json,
# candidates2.json and notebook.json. Override via the EXP1_DIR env var.
EXP1_DIR = os.environ.get("EXP1_DIR", "../exp1-scar")
sys.path.insert(0, EXP1_DIR)
import cd_lib as L  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# Re-export of identical core functions
prompt_plain = L.prompt_plain
prompt_single_fact = L.prompt_single_fact
prompt_notebook = L.prompt_notebook
per_token_logps = L.per_token_logps
two_way = L.two_way
gain = L.gain
contrastive_generate = L.contrastive_generate
SYS_PLAIN = L.SYS_PLAIN
SYS_NB = L.SYS_NB
ALPHA = L.ALPHA
BETA = L.BETA

torch.set_grad_enabled(False)

_END_TOKENS = ("<|im_end|>", "<|end|>", "<|eot_id|>", "<|endoftext|>", "<end_of_turn>", "<|eot|>")


def load_any(model_id):
    # trust_remote_code=False: for Phi-3.5 on transformers 5.x the built-in support is required
    # (the upstream modeling_phi3.py calls DynamicCache.from_legacy_cache, which was removed in 5.x).
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda").eval()
    stop_ids = {tok.eos_token_id}
    for tt in _END_TOKENS:
        tid = tok.convert_tokens_to_ids(tt)
        if isinstance(tid, int) and tid >= 0:
            stop_ids.add(tid)
    return tok, model, stop_ids
