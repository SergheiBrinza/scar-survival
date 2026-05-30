#!/usr/bin/env python3
"""One scar-survival cycle for (model, set) - each separate process = full reload.

Usage: python3 scar_cycle_multi.py <model_id> <setA|setB> <cycle_num>
Reads notebook_<modelshort>_<set>.json (this cell's working notebook), writes cycle_<...>_<n>.json.
"""
import os
import sys
import json
import cd_lib_mm as L

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = sys.argv[1]
SET = sys.argv[2]
CYCLE = int(sys.argv[3])
SHORT = MODEL.split("/")[-1].replace(".", "_")


def main():
    nb = json.load(open(os.path.join(HERE, f"notebook_{SHORT}_{SET}.json")))
    traps = nb["traps"]  # [{id, question, authoritative_fact, correct_answer}]
    print(f"[cycle {CYCLE}] {MODEL} / {SET}: loading...", flush=True)
    tok, model, stop = L.load_any(MODEL)

    rows = []
    for t in traps:
        i, q = t["id"], t["question"]
        fact = t["authoritative_fact"]
        p_with = L.prompt_single_fact(tok, fact, q)
        p_without = L.prompt_plain(tok, q)
        ans = L.contrastive_generate(model, tok, p_with, p_without, stop, greedy=True)
        rows.append({"id": i, "question": q,
                     "gold_fact": fact, "correct_answer": t["correct_answer"], "answer": ans})

    out = {"cycle": CYCLE, "model": MODEL, "set": SET, "rows": rows}
    json.dump(out, open(os.path.join(HERE, f"cycle_{SHORT}_{SET}_{CYCLE}.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[cycle {CYCLE}] {SHORT}/{SET}: done, {len(rows)} traps", flush=True)


if __name__ == "__main__":
    main()
