#!/usr/bin/env python3
"""Load 3: one stochastic cycle (separate process = full reload).
temperature=0.7, K samples per trap (generation is random). Confidence Gain is
deterministic (a property of the distributions) - we compute it once for reference.
Usage: python3 scar_stochastic_cycle.py <cycle> [K]
Writes stoch_cycle_<n>.json. alpha / beta parameters do not change.
"""
import os
import sys
import json
import cd_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
cycle = int(sys.argv[1])
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TEMP = 0.7

CANDS = {
    13: ("weeks to months", "a few seconds"), 16: ("the entire tongue", "the tip of the tongue"),
    20: ("Mauna Kea", "Mount Everest"), 23: ("Leif Erikson", "Christopher Columbus"),
    27: ("No, I am your father", "Luke, I am your father"), 28: ("Magic mirror on the wall", "Mirror, mirror on the wall"),
    29: ("Play it, Sam", "Play it again, Sam"), 34: ("Juan Sebastian Elcano", "Ferdinand Magellan"),
    35: ("Aristarchus of Samos", "Nicolaus Copernicus"), 36: ("Valentina Tereshkova", "Sally Ride"),
    37: ("fruit flies", "Laika the dog"), 38: ("Hans Lippershey", "Galileo Galilei"),
    40: ("The Adventures of Prince Achmed", "Snow White"), 46: ("Yes", "No"),
}


def main():
    nb = {e["id"]: e for e in json.load(open(os.path.join(HERE, "notebook.json")))["entries"]}
    traps = json.load(open(os.path.join(HERE, "working_set_3b.json")))["traps"]
    tok, model, stop = L.load()

    rows = []
    for t in traps:
        i, q = t["id"], t["question"]
        fact = nb[i]["authoritative_fact"]
        p_with = L.prompt_single_fact(tok, fact, q)
        p_without = L.prompt_plain(tok, q)
        corr, wrong = CANDS[i]
        _, _, _, g = L.gain(model, tok, p_with, p_without, corr, wrong)
        samples = []
        for k in range(K):
            seed = cycle * 10000 + i * 100 + k
            ans = L.contrastive_generate(model, tok, p_with, p_without, stop,
                                         greedy=False, temp=TEMP, seed=seed)
            samples.append(ans)
        rows.append({"id": i, "question": q,
                     "gold_fact": fact, "confidence_gain": round(g, 4), "samples": samples})
        print(f"[cycle {cycle}] #{i:>2} gain={g:+.3f} K={K}", flush=True)

    json.dump({"cycle": cycle, "temp": TEMP, "K": K, "rows": rows},
              open(os.path.join(HERE, f"stoch_cycle_{cycle}.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[cycle {cycle}] saved stoch_cycle_{cycle}.json", flush=True)


if __name__ == "__main__":
    main()
