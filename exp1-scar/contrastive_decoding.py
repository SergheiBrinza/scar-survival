#!/usr/bin/env python3
"""Sub-step 5b: CK-PLUG / DeCK style contrastive decoding on Qwen2.5-3B.

Method core: at each generation step we compare the token distribution WITH the
correct fact in context and WITHOUT it, then shift selection toward the fact:
    adj_logprob = (1+alpha) * logp_with_fact - alpha * logp_without_fact
plus a plausibility constraint (contrastive decoding): only consider tokens
whose logp_with >= max(logp_with) + log(beta), so that we do not amplify garbage.

Measurable indicator (in the spirit of CK-PLUG's Confidence Gain): for each trap
take the correct and the memorized answers as candidates and compute the share
of probability mass P(correct) in three regimes. Confidence Gain = P_contrastive - P_instinct.

Model weights are not touched (frozen). Saves contrastive_results.json.
"""
import os
import math
import json
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-3B-Instruct"
ALPHA = 1.0   # contrast strength
BETA = 0.1    # plausibility threshold

SYS_PLAIN = "You are a helpful assistant. Answer the question directly and concisely."
SYS_NB = ("You are a helpful assistant. Answer the question directly and concisely. "
          "An authoritative, verified fact is provided. Treat it as ground truth and "
          "base your answer on it, even if it conflicts with what you remember.")

# (correct, memorized) short candidates used to measure the mass shift
CANDS = {
    13: ("weeks to months", "a few seconds"),
    16: ("the entire tongue", "the tip of the tongue"),
    20: ("Mauna Kea", "Mount Everest"),
    23: ("Leif Erikson", "Christopher Columbus"),
    27: ("No, I am your father", "Luke, I am your father"),
    28: ("Magic mirror on the wall", "Mirror, mirror on the wall"),
    29: ("Play it, Sam", "Play it again, Sam"),
    34: ("Juan Sebastian Elcano", "Ferdinand Magellan"),
    35: ("Aristarchus of Samos", "Nicolaus Copernicus"),
    36: ("Valentina Tereshkova", "Sally Ride"),
    37: ("fruit flies", "Laika the dog"),
    38: ("Hans Lippershey", "Galileo Galilei"),
    40: ("The Adventures of Prince Achmed", "Snow White"),
    46: ("Yes", "No"),
}

torch.set_grad_enabled(False)


def chat_ids(tok, system, user):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, dict):
        out = out["input_ids"]
    return out


def per_token_logps(model, prompt_ids, cand_ids):
    full = torch.cat([prompt_ids, torch.tensor([cand_ids])], 1).to(model.device)
    logits = model(full).logits[0].float()
    lp = torch.log_softmax(logits, -1)
    plen = prompt_ids.shape[1]
    return [lp[plen - 1 + j, t].item() for j, t in enumerate(cand_ids)]


def two_way(score_c, score_w):
    m = max(score_c, score_w)
    ec, ew = math.exp(score_c - m), math.exp(score_w - m)
    return ec / (ec + ew)


def contrastive_generate(model, ids_with, ids_without, stop_ids, max_new=64):
    dev = model.device
    iw = ids_with.to(dev).clone()
    inp = ids_without.to(dev).clone()
    gen = []
    for _ in range(max_new):
        lw = torch.log_softmax(model(iw).logits[0, -1].float(), -1)
        ln = torch.log_softmax(model(inp).logits[0, -1].float(), -1)
        mask = lw < (lw.max() + math.log(BETA))
        adj = (1 + ALPHA) * lw - ALPHA * ln
        adj = adj.masked_fill(mask, float("-inf"))
        tid = int(adj.argmax())
        if tid in stop_ids:
            break
        gen.append(tid)
        nt = torch.tensor([[tid]], device=dev)
        iw = torch.cat([iw, nt], 1)
        inp = torch.cat([inp, nt], 1)
    return gen


def main():
    nb = {e["id"]: e for e in json.load(open(os.path.join(HERE, "notebook.json")))["entries"]}
    traps = json.load(open(os.path.join(HERE, "working_set_3b.json")))["traps"]

    print(f"[5b] Loading {MODEL} (transformers) on GPU 0 ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()

    stop_ids = {tok.eos_token_id}
    imend = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(imend, int) and imend >= 0:
        stop_ids.add(imend)

    rows = []
    for t in traps:
        i, q = t["id"], t["question"]
        fact = nb[i]["authoritative_fact"]
        p_with = chat_ids(tok, SYS_NB, f"Authoritative fact: {fact}\n\nQuestion: {q}")
        p_without = chat_ids(tok, SYS_PLAIN, q)

        # --- metric: probability mass shift (correct vs memorized) ---
        corr, wrong = CANDS[i]
        ci = tok(corr, add_special_tokens=False).input_ids
        wi = tok(wrong, add_special_tokens=False).input_ids
        cw = per_token_logps(model, p_with, ci)      # correct, with fact
        cn = per_token_logps(model, p_without, ci)   # correct, no fact
        ww = per_token_logps(model, p_with, wi)      # wrong, with fact
        wn = per_token_logps(model, p_without, wi)   # wrong, no fact

        def mean(xs):
            return sum(xs) / len(xs)

        def contr(lps_with, lps_no):
            return mean([(1 + ALPHA) * a - ALPHA * b for a, b in zip(lps_with, lps_no)])

        P_instinct = two_way(mean(cn), mean(wn))
        P_context = two_way(mean(cw), mean(ww))
        P_contrastive = two_way(contr(cw, cn), contr(ww, wn))
        gain = P_contrastive - P_instinct

        # --- the actual answer under contrastive decoding ---
        gen_ids = contrastive_generate(model, p_with, p_without, stop_ids)
        answer = tok.decode(gen_ids, skip_special_tokens=True).strip()

        rows.append({
            "id": i, "question": q,
            "gold_fact": fact, "correct_answer": t["correct_answer"],
            "cand_correct": corr, "cand_wrong": wrong,
            "P_instinct": round(P_instinct, 4),
            "P_context": round(P_context, 4),
            "P_contrastive": round(P_contrastive, 4),
            "confidence_gain": round(gain, 4),
            "answer_contrastive": answer,
        })
        print(f"\n#{i:>2}  P_inst={P_instinct:.3f}  P_ctx={P_context:.3f}  "
              f"P_contr={P_contrastive:.3f}  gain={gain:+.3f}")
        print(f"     contrastive answer: {answer}")

    out = {"model": MODEL, "method": "contrastive decoding (CK-PLUG/DeCK style)",
           "alpha": ALPHA, "beta": BETA,
           "metric": "P(correct) two-way mass; Confidence Gain = P_contrastive - P_instinct",
           "rows": rows}
    json.dump(out, open(os.path.join(HERE, "contrastive_results.json"), "w"),
              ensure_ascii=False, indent=2)
    avg_gain = sum(r["confidence_gain"] for r in rows) / len(rows)
    print("\n" + "=" * 60)
    print(f"Average Confidence Gain across 14 traps: {avg_gain:+.3f}")
    print("Saved contrastive_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
