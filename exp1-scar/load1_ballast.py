#!/usr/bin/env python3
"""Load 1: growing notebook. Correct fact + K unrelated entries in context.
Levels K = 0, 50, 100, 200, 500. For each: gain + contrastive answer over 14 traps.
Writes load1_answers.json. alpha / beta parameters do not change.
"""
import os
import json
import random
import cd_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = [0, 50, 100, 200, 500]
CANDS = {  # (correct, wrong) for the metric
    13: ("weeks to months", "a few seconds"), 16: ("the entire tongue", "the tip of the tongue"),
    20: ("Mauna Kea", "Mount Everest"), 23: ("Leif Erikson", "Christopher Columbus"),
    27: ("No, I am your father", "Luke, I am your father"), 28: ("Magic mirror on the wall", "Mirror, mirror on the wall"),
    29: ("Play it, Sam", "Play it again, Sam"), 34: ("Juan Sebastian Elcano", "Ferdinand Magellan"),
    35: ("Aristarchus of Samos", "Nicolaus Copernicus"), 36: ("Valentina Tereshkova", "Sally Ride"),
    37: ("fruit flies", "Laika the dog"), 38: ("Hans Lippershey", "Galileo Galilei"),
    40: ("The Adventures of Prince Achmed", "Snow White"), 46: ("Yes", "No"),
}


def make_ballast(n):
    random.seed(0)
    cities = ["Vienna", "Lima", "Osaka", "Cairo", "Oslo", "Perth", "Quito", "Riga", "Accra", "Bern"]
    out = []
    for k in range(n):
        c = random.choice(cities); code = random.randint(1000, 9999)
        sec = random.choice("ABCDEFGH"); cab = random.randint(1, 40)
        out.append(f"Archive note {k:04d}: catalog parcel {code} from the {c} office "
                   f"was filed under section {sec}-{cab}.")
    return out


def main():
    nb = {e["id"]: e for e in json.load(open(os.path.join(HERE, "notebook.json")))["entries"]}
    traps = json.load(open(os.path.join(HERE, "working_set_3b.json")))["traps"]
    ballast_pool = make_ballast(max(LEVELS))

    tok, model, stop = L.load()
    rows = []
    for lvl in LEVELS:
        ballast = ballast_pool[:lvl]
        for t in traps:
            i, q = t["id"], t["question"]
            fact = nb[i]["authoritative_fact"]
            facts = [fact] + ballast
            rnd = random.Random(1000 + i + lvl)
            rnd.shuffle(facts)
            p_with = L.prompt_notebook(tok, facts, q)
            p_without = L.prompt_plain(tok, q)
            corr, wrong = CANDS[i]
            P_inst, P_ctx, P_contr, g = L.gain(model, tok, p_with, p_without, corr, wrong)
            ans = L.contrastive_generate(model, tok, p_with, p_without, stop, greedy=True)
            rows.append({"level": lvl, "id": i, "question": q,
                         "gold_fact": fact, "P_instinct": round(P_inst, 4),
                         "P_contrastive": round(P_contr, 4), "confidence_gain": round(g, 4),
                         "answer": ans})
        avgg = sum(r["confidence_gain"] for r in rows if r["level"] == lvl) / len(traps)
        print(f"[level {lvl:>3}] avg gain={avgg:+.3f}", flush=True)

    json.dump({"load": "1_ballast", "levels": LEVELS, "rows": rows},
              open(os.path.join(HERE, "load1_answers.json"), "w"), ensure_ascii=False, indent=2)
    print("Saved load1_answers.json")


if __name__ == "__main__":
    main()
