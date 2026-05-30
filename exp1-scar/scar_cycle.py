#!/usr/bin/env python3
"""ONE scar-survival cycle, run as a SEPARATE process (full unload/reload).
Usage:  python3 scar_cycle.py <cycle_number>
Fresh load of 3B from disk -> reload of the notebook from disk -> contrastive decoding over 14 traps.
Parameters do NOT change between cycles (taken from contrastive_decoding: ALPHA, BETA).
Writes cycle_<n>.json. Weights are frozen; nothing is tuned.
"""
import os
import sys
import json
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from transformers import AutoModelForCausalLM, AutoTokenizer
import contrastive_decoding as C  # reuse the method core and its parameters

HERE = os.path.dirname(os.path.abspath(__file__))
cycle = int(sys.argv[1])
MODEL = C.MODEL

torch.set_grad_enabled(False)


def mean(xs):
    return sum(xs) / len(xs)


def contr(lps_with, lps_no):
    return mean([(1 + C.ALPHA) * a - C.ALPHA * b for a, b in zip(lps_with, lps_no)])


def main():
    print(f"[cycle {cycle}] fresh load of {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
    stop_ids = {tok.eos_token_id}
    imend = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(imend, int) and imend >= 0:
        stop_ids.add(imend)

    # the notebook is reloaded from disk on each cycle
    nb = {e["id"]: e for e in json.load(open(os.path.join(HERE, "notebook.json")))["entries"]}
    traps = json.load(open(os.path.join(HERE, "working_set_3b.json")))["traps"]

    rows = []
    for t in traps:
        i, q = t["id"], t["question"]
        fact = nb[i]["authoritative_fact"]
        p_with = C.chat_ids(tok, C.SYS_NB, f"Authoritative fact: {fact}\n\nQuestion: {q}")
        p_without = C.chat_ids(tok, C.SYS_PLAIN, q)

        corr, wrong = C.CANDS[i]
        ci = tok(corr, add_special_tokens=False).input_ids
        wi = tok(wrong, add_special_tokens=False).input_ids
        cw = C.per_token_logps(model, p_with, ci)
        cn = C.per_token_logps(model, p_without, ci)
        ww = C.per_token_logps(model, p_with, wi)
        wn = C.per_token_logps(model, p_without, wi)

        P_instinct = C.two_way(mean(cn), mean(wn))
        P_contrastive = C.two_way(contr(cw, cn), contr(ww, wn))
        gain = P_contrastive - P_instinct

        gen_ids = C.contrastive_generate(model, p_with, p_without, stop_ids)
        answer = tok.decode(gen_ids, skip_special_tokens=True).strip()

        rows.append({
            "id": i, "question": q,
            "gold_fact": fact,
            "P_instinct": round(P_instinct, 4),
            "P_contrastive": round(P_contrastive, 4),
            "confidence_gain": round(gain, 4),
            "answer_contrastive": answer,
        })
        print(f"  #{i:>2} gain={gain:+.3f}  {answer[:60]}", flush=True)

    out = {"cycle": cycle, "model": MODEL, "alpha": C.ALPHA, "beta": C.BETA, "rows": rows}
    json.dump(out, open(os.path.join(HERE, f"cycle_{cycle}.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[cycle {cycle}] saved cycle_{cycle}.json", flush=True)


if __name__ == "__main__":
    main()
