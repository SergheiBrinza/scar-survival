#!/usr/bin/env python3
"""Baseline 'instinct' measurement for ANY model across ALL 58 candidates
(facts.json 1-30 + candidates2.json 31-58). english, greedy (temperature=0), clean.
Usage:  python3 run_model.py "Qwen/Qwen2.5-3B-Instruct"
Saves result_<model>.json. Model weights are not touched.
"""
import os
import re
import sys
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"

MARKERS = {
    1:  {"correct": ["canberra"], "wrong": ["sydney"]},
    2:  {"correct": ["ankara"], "wrong": ["istanbul"]},
    3:  {"correct": ["ottawa"], "wrong": ["toronto"]},
    4:  {"correct": ["brasilia", "brasília"], "wrong": ["rio de janeiro", "rio"]},
    5:  {"correct": ["washington"], "wrong": ["new york"]},
    6:  {"correct": ["bern"], "wrong": ["zurich", "zürich", "geneva"]},
    7:  {"correct": ["eight", "8"], "wrong": ["nine", "9"]},
    8:  {"correct": ["red"], "wrong": ["blue"]},
    9:  {"correct": ["white"], "wrong": ["yellow"]},
    10: {"correct": ["all of", "entire brain", "whole brain", "virtually all", "most of", "myth", "100%", "use all", "more than 10", "not just 10"], "wrong": ["only 10", "just 10", "10% of", "10 percent", "use 10", "about 10%", "ten percent"]},
    11: {"correct": ["not visible", "no,", "cannot", "can't", "myth", "false", "isn't visible", "is not visible", "not actually"], "wrong": ["yes", "is visible", "can be seen", "visible from"]},
    12: {"correct": ["not blind", "can see", "aren't blind", "are not blind", "myth", "good eyesight", "see well", "not actually blind", "do see"], "wrong": ["are blind", "is blind", "yes", "blind"]},
    13: {"correct": ["months", "weeks", "much longer", "not three", "not 3", "myth", "long-term", "remember for"], "wrong": ["3 second", "three second", "few seconds", "seconds", "short"]},
    14: {"correct": ["pass", "days", "not seven", "not 7", "myth", "normal", "digestive", "excret", "through your system"], "wrong": ["seven years", "7 years"]},
    15: {"correct": ["can strike", "does strike", "yes", "often", "repeatedly", "multiple times", "frequently", "it can", "more than once"], "wrong": ["never", "cannot", "can't", "does not strike", "won't"]},
    16: {"correct": ["all parts", "whole tongue", "entire tongue", "all areas", "everywhere", "myth", "not just the tip", "no single", "across the tongue"], "wrong": ["only the tip", "just the tip", "the tip of", "front of the tongue"]},
    17: {"correct": ["poor", "does not conduct", "doesn't conduct", "not a good conductor", "bad conductor", "insulator", "not conduct", "poorly", "no,"], "wrong": ["yes", "good conductor", "conducts electricity", "is a good"]},
    18: {"correct": ["no,", "does not flow", "doesn't flow", "myth", "amorphous solid", "is a solid", "not a liquid", "manufacturing", "false", "not flow"], "wrong": ["yes", "slowly flows", "is a liquid", "flows over", "does flow"]},
    19: {"correct": ["tilt", "axis", "axial", "tilted"], "wrong": ["distance", "closer to the sun", "farther", "how close"]},
    20: {"correct": ["mauna kea"], "wrong": ["everest"]},
    21: {"correct": ["no,", "did not fail", "didn't fail", "excelled", "good at math", "myth", "false", "never failed", "strong in math", "mastered", "did well"], "wrong": ["yes", "failed math", "he failed", "did fail"]},
    22: {"correct": ["no,", "average", "not short", "myth", "normal height", "wasn't short", "misconception", "taller", "not unusually", "about average"], "wrong": ["yes", "very short", "was short", "quite short", "5 feet 2", "unusually short", "considered short"]},
    23: {"correct": ["leif", "erikson", "ericson", "norse", "viking"], "wrong": ["columbus"]},
    24: {"correct": ["no,", "did not", "didn't", "no evidence", "myth", "false", "not wear", "no horns", "without horns", "19th century", "did not wear"], "wrong": ["yes", "horned", "wore horns", "did wear"]},
    25: {"correct": ["no,", "did not", "didn't", "no evidence", "myth", "misattribut", "never said", "unlikely", "predates", "probably did not", "no record"], "wrong": ["yes", "she said", "did say"]},
    26: {"correct": ["hang", "hung", "pressed"], "wrong": ["burn", "stake"]},
    27: {"correct": ["no, i am your father", "no, i'm your father", "misquote", "misquotation", "never says luke", "does not say luke", "actually", "commonly misquoted"], "wrong": ["luke, i am your father", "luke, i'm your father"]},
    28: {"correct": ["magic mirror"], "wrong": ["mirror, mirror", "mirror mirror"]},
    29: {"correct": ["play it, sam", "play it once", "never said", "misquote", "not actually", "misquotation"], "wrong": ["play it again, sam"]},
    30: {"correct": ["love of money"], "wrong": ["money is the root of all evil"]},
    31: {"correct": ["116", "more than 100", "not 100"], "wrong": ["100 years", "exactly 100"]},
    32: {"correct": ["1773"], "wrong": ["1776", "1775", "1774", "1770"]},
    33: {"correct": ["amundsen"], "wrong": ["scott"]},
    34: {"correct": ["elcano", "del cano"], "wrong": ["magellan"]},
    35: {"correct": ["aristarchus"], "wrong": ["copernicus", "galileo"]},
    36: {"correct": ["tereshkova"], "wrong": ["sally ride", "ride"]},
    37: {"correct": ["fruit fl"], "wrong": ["laika", "the dog", "a dog"]},
    38: {"correct": ["lippershey", "lipperhey", "lippernay"], "wrong": ["galileo"]},
    39: {"correct": ["bi sheng", "china", "chinese", "korea"], "wrong": ["gutenberg"]},
    40: {"correct": ["prince achmed", "achmed", "apostol", "apóstol", "1926", "1917"], "wrong": ["snow white", "1937", "pinocchio", "1940", "fantasia", "bambi", "dumbo", "1942"]},
    41: {"correct": ["venus"], "wrong": ["mercury"]},
    42: {"correct": ["nitrogen"], "wrong": ["oxygen", "carbon dioxide", "co2"]},
    43: {"correct": ["three"], "wrong": ["one", "two", "single", "1 heart"]},
    44: {"correct": ["compartment", "chamber", "one stomach", "single stomach", "one large"], "wrong": ["four stomachs", "4 stomachs", "four separate"]},
    45: {"correct": ["mercury"], "wrong": ["pluto", "mars", "ceres"]},
    46: {"correct": ["yes", "longer than its year", "day is longer", "243", "longer than a year", "day on venus is longer"], "wrong": ["no,", "shorter", "year is longer"]},
    47: {"correct": ["300", "270", "around 300", "about 300"], "wrong": ["206"]},
    48: {"correct": ["mandarin", "chinese"], "wrong": ["english", "spanish", "hindi"]},
    49: {"correct": ["dog", "canis", "the bird is named", "named after the dog", "no,"], "wrong": ["yes", "named after the canary", "after the bird", "after the canaries"]},
    50: {"correct": ["iceland is", "actually iceland", "iceland, despite", "iceland has more", "answer is iceland"], "wrong": ["greenland is", "actually greenland", "greenland, despite", "greenland has more", "answer is greenland"]},
    51: {"correct": ["rodent", "south america", "andes", "neither", "not pigs", "not from guinea", "no,"], "wrong": ["yes", "are pigs", "from guinea", "type of pig"]},
    52: {"correct": ["beetle", "not a fly", "not flies", "no,"], "wrong": ["yes", "is a fly", "are flies", "type of fly"]},
    53: {"correct": ["india", "indian", "hindu", "not invented by the arab", "no,"], "wrong": ["yes", "invented by the arab", "invented by arab", "by the arabs"]},
    54: {"correct": ["ulnar", "nerve", "not a bone", "no,"], "wrong": ["yes", "is a bone", "actually a bone"]},
    55: {"correct": ["ecuador"], "wrong": ["panama"]},
    56: {"correct": ["north america", "americas", "not native to turkey", "new world", "no,"], "wrong": ["yes", "native to turkey", "from turkey", "in turkey"]},
    57: {"correct": ["sheep", "goat", "intestine", "not from cat", "not made from cat", "no,"], "wrong": ["yes", "from cats", "made from cat"]},
    58: {"correct": ["legume", "not a true nut", "not a nut", "not actually a nut", "no,"], "wrong": ["yes", "is a nut", "true nut", "are nuts"]},
}


def has(marker: str, text: str) -> bool:
    m = marker.lower()
    if len(m) <= 3 and m.isalnum():
        return re.search(r"\b" + re.escape(m) + r"\b", text) is not None
    return m in text


def classify(fact_id: int, answer: str) -> str:
    text = answer.lower()
    mk = MARKERS[fact_id]
    if any(has(c, text) for c in mk["correct"]):
        return "correct"
    if any(has(w, text) for w in mk["wrong"]):
        return "wrong_instinct"
    return "ambiguous"


def main():
    facts = json.load(open(os.path.join(HERE, "facts.json")))["facts"]
    facts += json.load(open(os.path.join(HERE, "candidates2.json")))["facts"]

    print(f"[run] Loading {MODEL} on GPU 0 (greedy, no notebook), {len(facts)} questions ...", flush=True)
    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=160)
    system = "You are a helpful assistant. Answer the question directly and concisely."

    results = []
    for f in facts:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f["question"]}]
        out = llm.chat(messages, params, use_tqdm=False)
        answer = out[0].outputs[0].text.strip()
        verdict = classify(f["id"], answer)
        results.append({
            "id": f["id"], "category": f["category"],
            "question": f["question"],
            "model_answer": answer, "verdict": verdict,
            "memorized_wrong_answer": f["memorized_wrong_answer"],
            "correct_answer": f["correct_answer"],
        })
        print(f"\n--- #{f['id']} [{verdict}] {f['question']}", flush=True)
        print(f"    model: {answer}", flush=True)

    fired = [r for r in results if r["verdict"] == "wrong_instinct"]
    not_fired = [r for r in results if r["verdict"] == "correct"]
    ambiguous = [r for r in results if r["verdict"] == "ambiguous"]
    total = len(results)
    rate = round(100.0 * len(fired) / total, 1)

    safe = MODEL.split("/")[-1].replace(".", "_")
    out = {"model": MODEL, "run_language": "english",
           "decoding": "greedy (temperature=0, seed=0)",
           "total": total, "n_fired": len(fired), "n_not_fired": len(not_fired),
           "n_ambiguous": len(ambiguous), "fire_rate_percent": rate,
           "fired_traps": fired, "not_fired": not_fired, "ambiguous": ambiguous,
           "results": results}
    outpath = os.path.join(HERE, f"result_{safe}.json")
    with open(outpath, "w") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"MODEL: {MODEL}")
    print(f"Fired (instinct won): {len(fired)}/{total} = {rate}%")
    print(f"  not fired: {len(not_fired)}   ambiguous: {len(ambiguous)}")
    print(f"Saved: {outpath}")
    print("=" * 60)


if __name__ == "__main__":
    main()
