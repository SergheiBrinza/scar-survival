#!/usr/bin/env python3
"""Reusable contrastive decoding core (CK-PLUG / DeCK style) for the stress test.
Supports: an arbitrary block of facts in context (ballast / noise), cached generation,
greedy and stochastic sampling. The alpha / beta parameters do NOT change between loads.
"""
import os
import math
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
ALPHA = 1.0
BETA = 0.1

SYS_PLAIN = "You are a helpful assistant. Answer the question directly and concisely."
SYS_NB = ("You are a helpful assistant. Answer the question directly and concisely. "
          "An authoritative, verified fact is provided. Treat it as ground truth and "
          "base your answer on it, even if it conflicts with what you remember.")

torch.set_grad_enabled(False)


def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
    stop_ids = {tok.eos_token_id}
    im = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im, int) and im >= 0:
        stop_ids.add(im)
    return tok, model, stop_ids


def _ids(tok, system, user):
    out = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        add_generation_prompt=True, return_tensors="pt")
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, dict):
        out = out["input_ids"]
    return out


def prompt_plain(tok, question):
    return _ids(tok, SYS_PLAIN, question)


def prompt_single_fact(tok, fact, question):
    return _ids(tok, SYS_NB, f"Authoritative fact: {fact}\n\nQuestion: {question}")


def prompt_notebook(tok, facts, question):
    """facts: list of strings (the correct fact + possible ballast / noise, already shuffled)."""
    block = "\n".join(f"- {f}" for f in facts)
    user = f"Notebook (authoritative verified facts):\n{block}\n\nQuestion: {question}"
    return _ids(tok, SYS_NB, user)


def per_token_logps(model, prompt_ids, cand_ids):
    full = torch.cat([prompt_ids, torch.tensor([cand_ids])], 1).to(model.device)
    lp = torch.log_softmax(model(full).logits[0].float(), -1)
    plen = prompt_ids.shape[1]
    return [lp[plen - 1 + j, t].item() for j, t in enumerate(cand_ids)]


def two_way(sc, sw):
    m = max(sc, sw)
    ec, ew = math.exp(sc - m), math.exp(sw - m)
    return ec / (ec + ew)


def gain(model, tok, p_with, p_without, correct_str, wrong_str):
    """Returns (P_instinct, P_context, P_contrastive, confidence_gain)."""
    ci = tok(correct_str, add_special_tokens=False).input_ids
    wi = tok(wrong_str, add_special_tokens=False).input_ids
    cw = per_token_logps(model, p_with, ci)
    cn = per_token_logps(model, p_without, ci)
    ww = per_token_logps(model, p_with, wi)
    wn = per_token_logps(model, p_without, wi)
    mean = lambda xs: sum(xs) / len(xs)
    contr = lambda a, b: mean([(1 + ALPHA) * x - ALPHA * y for x, y in zip(a, b)])
    P_inst = two_way(mean(cn), mean(wn))
    P_ctx = two_way(mean(cw), mean(ww))
    P_contr = two_way(contr(cw, cn), contr(ww, wn))
    return P_inst, P_ctx, P_contr, P_contr - P_inst


def contrastive_generate(model, tok, p_with, p_without, stop_ids,
                         max_new=64, greedy=True, temp=0.7, seed=0):
    """Contrastive generation with a KV cache. greedy=True -> argmax; else sampling with temp."""
    dev = model.device
    ow = model(p_with.to(dev), use_cache=True)
    on = model(p_without.to(dev), use_cache=True)
    pw, pn = ow.past_key_values, on.past_key_values
    lw_last, ln_last = ow.logits[:, -1], on.logits[:, -1]
    g = torch.Generator(device=dev); g.manual_seed(seed)
    gen = []
    for _ in range(max_new):
        lw = torch.log_softmax(lw_last[0].float(), -1)
        ln = torch.log_softmax(ln_last[0].float(), -1)
        mask = lw < (lw.max() + math.log(BETA))
        adj = ((1 + ALPHA) * lw - ALPHA * ln).masked_fill(mask, float("-inf"))
        if greedy:
            tid = int(adj.argmax())
        else:
            probs = torch.softmax(adj / temp, -1)
            tid = int(torch.multinomial(probs, 1, generator=g))
        if tid in stop_ids:
            break
        gen.append(tid)
        nt = torch.tensor([[tid]], device=dev)
        ow = model(nt, past_key_values=pw, use_cache=True); pw = ow.past_key_values; lw_last = ow.logits[:, -1]
        on = model(nt, past_key_values=pn, use_cache=True); pn = on.past_key_values; ln_last = on.logits[:, -1]
    return tok.decode(gen, skip_special_tokens=True).strip()
