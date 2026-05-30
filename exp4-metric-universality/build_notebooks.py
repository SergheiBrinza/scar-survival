#!/usr/bin/env python3
"""Build per-cell notebooks from picks_*.json.

For each cell (model x set) we assemble a notebook with the authoritative fact for
each selected trap.

Fact sources:
  - id in experiment-1 notebook.json (14 working traps) -> reuse its authoritative_fact
  - id in 1-58 (experiment-1 candidates2/facts) without notebook -> synthesised from correct_answer
  - id in 100-119 (Set B) -> authoritative_fact from setB_candidates.json
Writes notebook_<modelshort>_<set>.json.
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
# Experiment 1 supplies the canonical notebook and Set A.
EXP1_DIR = os.environ.get("EXP1_DIR", os.path.join(HERE, "..", "exp1-scar"))

MODELS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3.5-mini-instruct"]
SETS = ["setA", "setB"]


def main():
    exp1_nb = {e["id"]: e for e in json.load(open(os.path.join(EXP1_DIR, "notebook.json")))["entries"]}
    setB = {e["id"]: e for e in json.load(open(os.path.join(HERE, "setB_candidates.json")))["facts"]}

    for M in MODELS:
        short = M.split("/")[-1].replace(".", "_")
        for S in SETS:
            picks_path = os.path.join(HERE, f"picks_{short}_{S}.json")
            if not os.path.exists(picks_path):
                print(f"skip {picks_path}")
                continue
            picks = json.load(open(picks_path))["picks"]
            traps = []
            for p in picks:
                i = p["id"]
                if i in exp1_nb:
                    fact = exp1_nb[i]["authoritative_fact"]
                elif i in setB:
                    fact = setB[i]["authoritative_fact"]
                else:
                    fact = f"Verified fact: {p['correct_answer']}"
                traps.append({"id": i, "question": p["question"],
                              "correct_answer": p["correct_answer"], "authoritative_fact": fact})
            nb = {"model": M, "set": S, "n_traps": len(traps), "traps": traps}
            json.dump(nb, open(os.path.join(HERE, f"notebook_{short}_{S}.json"), "w"),
                      ensure_ascii=False, indent=2)
            print(f"notebook_{short}_{S}.json: {len(traps)} working traps")


if __name__ == "__main__":
    main()
