#!/usr/bin/env python3
"""One scar-survival cycle in one of THREE regimes on Qwen2.5-3B (full reload).
Usage: python3 discriminate_cycle.py <R1|R2|R3> <cycle_num>

R1 - REAL scar: notebook is reconnected every cycle, contrastive with the correct fact.
R2 - COSMETIC: cycle 1 - contrastive with the correct fact, cycles 2-10 - WITHOUT notebook (plain instinct).
R3 - PLACEBO: notebook is reconnected every cycle but contains an IRRELEVANT entry (contrastive idles).
"""
import os
import sys
import json

OP1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exp1-scar"))
sys.path.insert(0, OP1)
import cd_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
REGIME = sys.argv[1]
CYCLE = int(sys.argv[2])
assert REGIME in {"R1", "R2", "R3"}


def main():
    real_nb = {e["id"]: e for e in json.load(open(os.path.join(OP1, "notebook.json")))["entries"]}
    placebo_nb = {e["id"]: e for e in json.load(open(os.path.join(HERE, "placebo_notebook.json")))["entries"]}
    traps = json.load(open(os.path.join(OP1, "working_set_3b.json")))["traps"]

    print(f"[{REGIME} cycle {CYCLE}] fresh Qwen2.5-3B load", flush=True)
    tok, model, stop = L.load()

    rows = []
    for t in traps:
        i, q = t["id"], t["question"]
        gold_fact_real = real_nb[i]["authoritative_fact"]
        p_plain = L.prompt_plain(tok, q)

        if REGIME == "R1":
            input_kind = "real_fact"
            p_with = L.prompt_single_fact(tok, gold_fact_real, q)
            p_without = p_plain
        elif REGIME == "R2":
            if CYCLE == 1:
                input_kind = "real_fact"
                p_with = L.prompt_single_fact(tok, gold_fact_real, q)
                p_without = p_plain
            else:
                input_kind = "none_plain_instinct"
                p_with = p_plain
                p_without = p_plain  # contrastive degenerates to pure instinct
        else:  # R3
            input_kind = "placebo"
            placebo_fact = placebo_nb[i]["authoritative_fact"]
            p_with = L.prompt_single_fact(tok, placebo_fact, q)
            p_without = p_plain

        ans = L.contrastive_generate(model, tok, p_with, p_without, stop, greedy=True)
        rows.append({"id": i, "question": q,
                     "gold_fact": gold_fact_real, "correct_answer": t["correct_answer"],
                     "regime": REGIME, "cycle": CYCLE, "input_kind": input_kind,
                     "answer": ans})

    out = {"regime": REGIME, "cycle": CYCLE, "model": "Qwen/Qwen2.5-3B-Instruct", "rows": rows}
    json.dump(out, open(os.path.join(HERE, f"disc_{REGIME}_{CYCLE}.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[{REGIME} cycle {CYCLE}] done, {len(rows)} traps", flush=True)


if __name__ == "__main__":
    main()
