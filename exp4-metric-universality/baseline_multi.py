#!/usr/bin/env python3
"""Baseline measurement (clean instinct, no notebook) for one model on one fact set.

Usage: python3 baseline_multi.py <model_id> <setA|setB>
Saves baseline_<modelshort>_<set>.json.
"""
import os
import sys
import json
import cd_lib_mm as L

HERE = os.path.dirname(os.path.abspath(__file__))
# Experiment 1 supplies Set A (facts.json + candidates2.json).
# Override EXP1_DIR if it lives elsewhere.
EXP1_DIR = os.environ.get("EXP1_DIR", os.path.join(HERE, "..", "exp1-scar"))
MODEL = sys.argv[1]
SET = sys.argv[2]


def load_set(name):
    if name == "setA":
        a = json.load(open(os.path.join(EXP1_DIR, "facts.json")))["facts"]
        b = json.load(open(os.path.join(EXP1_DIR, "candidates2.json")))["facts"]
        return a + b
    if name == "setB":
        return json.load(open(os.path.join(HERE, "setB_candidates.json")))["facts"]
    raise ValueError(name)


def main():
    cands = load_set(SET)
    tok, model, stop = L.load_any(MODEL)
    print(f"[baseline] {MODEL} on {SET}: {len(cands)} questions", flush=True)
    rows = []
    for c in cands:
        i, q = c["id"], c["question"]
        p = L.prompt_plain(tok, q)
        # clean instinct = contrastive with p_with == p_without
        ans = L.contrastive_generate(model, tok, p, p, stop, greedy=True)
        gold = c.get("correct_answer") or c.get("authoritative_fact")
        rows.append({"id": i, "question": q,
                     "memorized_wrong_answer": c.get("memorized_wrong_answer", ""),
                     "correct_answer": gold, "instinct_answer": ans})
        print(f"  #{i:>3} {ans[:60]}", flush=True)

    short = MODEL.split("/")[-1].replace(".", "_")
    out = {"model": MODEL, "set": SET, "n": len(rows), "rows": rows}
    json.dump(out, open(os.path.join(HERE, f"baseline_{short}_{SET}.json"), "w"), ensure_ascii=False, indent=2)
    print(f"Saved baseline_{short}_{SET}.json")


if __name__ == "__main__":
    main()
