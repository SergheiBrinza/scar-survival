#!/usr/bin/env python3
"""Baseline 'instinct' measurement on NEW candidates (candidates2.json), id 31-58.
Same conditions: english, greedy (temperature=0), clean, no hints.
Saves candidates_result.json. Model weights are not touched.
"""
import os
import re
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))

# correct-markers have priority; for yes/no questions the distinctive 'true token'
# is listed first, so that a myth-debunking answer is not counted as an error.
MARKERS = {
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
    facts = json.load(open(os.path.join(HERE, "candidates2.json")))["facts"]

    print(f"[candidates] Loading {MODEL} on GPU 0 (greedy, no notebook) ...", flush=True)
    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=160)
    system = "You are a helpful assistant. Answer the question directly and concisely."

    results = []
    for f in facts:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f["question"]},
        ]
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
    hit_rate = round(100.0 * len(fired) / total, 1)

    out = {
        "model": MODEL, "run_language": "english",
        "decoding": "greedy (temperature=0, seed=0)",
        "total": total, "n_fired": len(fired), "n_not_fired": len(not_fired),
        "n_ambiguous": len(ambiguous), "hit_rate_percent": hit_rate,
        "fired_traps": fired, "not_fired": not_fired, "ambiguous": ambiguous,
        "results": results,
    }
    with open(os.path.join(HERE, "candidates_result.json"), "w") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"NEW CANDIDATES: fired {len(fired)}/{total} = {hit_rate}%")
    print(f"  not fired: {len(not_fired)}   ambiguous: {len(ambiguous)}")
    print("=" * 60)
    print("Saved to candidates_result.json")


if __name__ == "__main__":
    main()
