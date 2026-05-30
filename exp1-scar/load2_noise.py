#!/usr/bin/env python3
"""Load 2: noisy / conflicting notebook. Alongside the correct fact we add
2-3 similar but MISLEADING entries (including the memorized wrong answer). Measures gain
and the contrastive answer, compared with the clean notebook (5b). Writes load2_answers.json.
"""
import os
import json
import random
import cd_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
CANDS = {
    13: ("weeks to months", "a few seconds"), 16: ("the entire tongue", "the tip of the tongue"),
    20: ("Mauna Kea", "Mount Everest"), 23: ("Leif Erikson", "Christopher Columbus"),
    27: ("No, I am your father", "Luke, I am your father"), 28: ("Magic mirror on the wall", "Mirror, mirror on the wall"),
    29: ("Play it, Sam", "Play it again, Sam"), 34: ("Juan Sebastian Elcano", "Ferdinand Magellan"),
    35: ("Aristarchus of Samos", "Nicolaus Copernicus"), 36: ("Valentina Tereshkova", "Sally Ride"),
    37: ("fruit flies", "Laika the dog"), 38: ("Hans Lippershey", "Galileo Galilei"),
    40: ("The Adventures of Prince Achmed", "Snow White"), 46: ("Yes", "No"),
}
# 2-3 MISLEADING (similar but wrong) entries per trap
DISTRACTORS = {
    13: ["A goldfish's memory lasts only about 3 seconds before it forgets.",
         "A goldfish can remember things for at most a few hours."],
    16: ["Sweet tastes are detected only at the tip of the tongue.",
         "The back of the tongue detects sweetness, the tip detects salt."],
    20: ["Measured from base to summit, the tallest mountain on Earth is Mount Everest.",
         "Measured from base to summit, the tallest mountain on Earth is K2."],
    23: ["The first European to reach the Americas was Christopher Columbus in 1492.",
         "The first European to reach the Americas was Amerigo Vespucci.",
         "The first European to reach the Americas was John Cabot in 1497."],
    27: ["In The Empire Strikes Back, Darth Vader says: 'Luke, I am your father.'",
         "Darth Vader's exact line is: 'Luke, I am your dad.'"],
    28: ["In Disney's Snow White, the Queen says: 'Mirror, mirror on the wall.'",
         "The Queen's line is: 'Mirror, mirror, on the wall, who is the fairest of them all.'"],
    29: ["In Casablanca, the famous line is: 'Play it again, Sam.'",
         "In Casablanca, Rick says: 'Play it again, Sam, for old times sake.'"],
    34: ["The first person to complete a circumnavigation of the globe was Ferdinand Magellan.",
         "The first person to circumnavigate the globe was Francis Drake."],
    35: ["The first person to propose that the Earth revolves around the Sun was Nicolaus Copernicus.",
         "The first to propose heliocentrism was Galileo Galilei."],
    36: ["The first woman in space was Sally Ride.",
         "The first woman in space was Svetlana Savitskaya."],
    37: ["The first animal ever sent into space was Laika the dog.",
         "The first animal in space was a monkey named Albert."],
    38: ["The telescope was invented by Galileo Galilei.",
         "The telescope was invented by Isaac Newton."],
    40: ["The oldest surviving feature-length animated film is Snow White (1937).",
         "The oldest surviving feature-length animated film is Steamboat Willie (1928)."],
    46: ["On Venus, a single day is shorter than its year.",
         "On Venus, a day and a year are about the same length."],
}


def main():
    nb = {e["id"]: e for e in json.load(open(os.path.join(HERE, "notebook.json")))["entries"]}
    traps = json.load(open(os.path.join(HERE, "working_set_3b.json")))["traps"]
    clean = {r["id"]: r for r in json.load(open(os.path.join(HERE, "contrastive_results.json")))["rows"]}

    tok, model, stop = L.load()
    rows = []
    for t in traps:
        i, q = t["id"], t["question"]
        fact = nb[i]["authoritative_fact"]
        facts = [fact] + DISTRACTORS[i]
        random.Random(2000 + i).shuffle(facts)
        p_with = L.prompt_notebook(tok, facts, q)
        p_without = L.prompt_plain(tok, q)
        corr, wrong = CANDS[i]
        P_inst, P_ctx, P_contr, g = L.gain(model, tok, p_with, p_without, corr, wrong)
        ans = L.contrastive_generate(model, tok, p_with, p_without, stop, greedy=True)
        rows.append({"id": i, "question": q, "gold_fact": fact,
                     "n_distractors": len(DISTRACTORS[i]),
                     "P_instinct": round(P_inst, 4), "P_contrastive": round(P_contr, 4),
                     "confidence_gain": round(g, 4),
                     "clean_gain": clean[i]["confidence_gain"],
                     "answer": ans})
        print(f"#{i:>2} noisy_gain={g:+.3f} (clean {clean[i]['confidence_gain']:+.3f})  {ans[:50]}", flush=True)

    json.dump({"load": "2_noise", "rows": rows},
              open(os.path.join(HERE, "load2_answers.json"), "w"), ensure_ascii=False, indent=2)
    print("Saved load2_answers.json")


if __name__ == "__main__":
    main()
