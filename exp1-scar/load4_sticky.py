#!/usr/bin/env python3
"""Load 4: sticky traps. Measure P_instinct over candidates, pick the most stubborn ones
(instinct close to 1.0 for the wrong answer), then run the full instinct/context/contrastive
chain + contrastive answer over them. Writes load4_answers.json + load4_all.json.
"""
import os
import json
import cd_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
THRESH = 0.10  # P_instinct(correct) below this = sticky

# (id, question, correct_str, wrong_str, authoritative_fact)
CAND = [
    ("s1", "What is the largest desert in the world?", "Antarctica", "the Sahara",
     "The largest desert in the world is Antarctica, a polar desert (~14 million km2). The Sahara is only the largest HOT desert."),
    ("s2", "What color is a polar bear's skin underneath its fur?", "black", "white",
     "A polar bear's skin is black; its fur is translucent and only appears white."),
    ("s3", "What color are flamingos when they first hatch?", "grey", "pink",
     "Flamingos hatch with grey plumage; they turn pink later from pigments in their diet."),
    ("s4", "What do camels store in their humps?", "fat", "water",
     "Camels store fat in their humps, not water."),
    ("s5", "Where does most of a tree's mass come from?", "the air", "the soil",
     "Most of a tree's mass comes from carbon dioxide in the air via photosynthesis, not from the soil."),
    ("s6", "Which mountain peak is the farthest point from the center of the Earth?", "Chimborazo", "Mount Everest",
     "Mount Chimborazo in Ecuador is the farthest point from Earth's center due to the equatorial bulge, though Everest is highest above sea level."),
    ("s7", "What is the actual full biblical quote about money and evil?", "the love of money", "money is the root of all evil",
     "The actual quote (1 Timothy 6:10) is 'the LOVE of money is the root of all evil', not 'money is the root of all evil'."),
    ("s8", "Do bulls become angry specifically at the color red?", "no", "yes",
     "Bulls are red-green colorblind; they charge at the cape because of its movement, not its red color."),
    ("s9", "Does the word 'sushi' mean raw fish?", "no", "yes",
     "'Sushi' refers to the vinegared rice, not raw fish; raw fish by itself is 'sashimi'."),
    ("s10", "Do diamonds form from coal?", "no", "yes",
     "Diamonds do not form from coal; they form from carbon under high pressure deep in the mantle, usually predating the plants that formed coal."),
    ("s11", "Why do chameleons primarily change color?", "communication", "to camouflage",
     "Chameleons change color mainly for communication and temperature regulation, not primarily to camouflage with their surroundings."),
    ("s12", "What is the official language of Brazil?", "Portuguese", "Spanish",
     "The official language of Brazil is Portuguese, not Spanish."),
    ("s13", "What is the official currency of Switzerland?", "Swiss franc", "Euro",
     "Switzerland uses the Swiss franc, not the Euro."),
    ("s14", "What is the most abundant metal in the Earth's crust?", "aluminium", "iron",
     "Aluminium is the most abundant metal in the Earth's crust, ahead of iron."),
    ("s15", "Do people swallow about eight spiders a year while sleeping?", "no", "yes",
     "The claim that people swallow several spiders a year in their sleep is a myth with no evidence."),
    ("s16", "Does the Coriolis effect determine which way water drains in a sink or toilet?", "no", "yes",
     "No - the Coriolis effect is far too weak at that scale; drain direction depends on the basin shape and how water enters."),
    ("s17", "Can a penny dropped from the Empire State Building kill a person below?", "no", "yes",
     "No - a penny's terminal velocity is too low to be lethal."),
    ("s18", "Do ostriches bury their heads in the sand when frightened?", "no", "yes",
     "No - ostriches do not bury their heads in the sand; that is a myth."),
    ("s19", "How many moons does Mercury have?", "zero", "one",
     "Mercury has zero moons."),
    ("s20", "Is the Great Wall of China a single continuous wall?", "no", "yes",
     "No - the Great Wall is a network of many separate walls and fortifications built over centuries, not one continuous wall."),
]


def main():
    tok, model, stop = L.load()
    all_rows = []
    for cid, q, corr, wrong, fact in CAND:
        p_plain = L.prompt_plain(tok, q)
        # P_instinct: pure instinct without the fact
        P_inst, _, _, _ = L.gain(model, tok, p_plain, p_plain, corr, wrong)
        all_rows.append({"cid": cid, "question": q, "correct_str": corr, "wrong_str": wrong,
                         "gold_fact": fact, "P_instinct": round(P_inst, 4)})
        print(f"{cid:>4} P_inst(correct)={P_inst:.3f}  | {q[:55]}", flush=True)

    all_rows.sort(key=lambda r: r["P_instinct"])
    sticky = [r for r in all_rows if r["P_instinct"] < THRESH]
    if len(sticky) < 10:
        sticky = all_rows[:12]
    sticky = sticky[:15]
    print(f"\nSelected sticky: {len(sticky)} (threshold P_inst<{THRESH})", flush=True)

    # over the selected ones - the full chain
    rows = []
    for r in sticky:
        q, corr, wrong, fact = r["question"], r["correct_str"], r["wrong_str"], r["gold_fact"]
        p_with = L.prompt_single_fact(tok, fact, q)
        p_without = L.prompt_plain(tok, q)
        P_inst, P_ctx, P_contr, g = L.gain(model, tok, p_with, p_without, corr, wrong)
        ans = L.contrastive_generate(model, tok, p_with, p_without, stop, greedy=True)
        rows.append({"id": r["cid"], "question": q, "gold_fact": fact,
                     "correct_str": corr, "wrong_str": wrong,
                     "P_instinct": round(P_inst, 4), "P_context": round(P_ctx, 4),
                     "P_contrastive": round(P_contr, 4), "confidence_gain": round(g, 4),
                     "answer": ans})
        print(f"{r['cid']:>4} inst={P_inst:.3f} ctx={P_ctx:.3f} contr={P_contr:.3f} gain={g:+.3f} | {ans[:45]}", flush=True)

    json.dump({"all_candidates": all_rows}, open(os.path.join(HERE, "load4_all.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump({"load": "4_sticky", "threshold": THRESH, "rows": rows},
              open(os.path.join(HERE, "load4_answers.json"), "w"), ensure_ascii=False, indent=2)
    avg_gain = sum(r["confidence_gain"] for r in rows) / len(rows)
    print(f"\nAverage Gain on sticky: {avg_gain:+.3f}. Saved load4_answers.json, load4_all.json")


if __name__ == "__main__":
    main()
