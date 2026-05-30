#!/usr/bin/env python3
"""Judges ALL answers from all 10 cycles in one pass with the independent judge Qwen2.5-7B.
Reads cycle_1.json..cycle_10.json -> writes all_cycles_judged.json (semantic verdicts).
"""
import os
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
JUDGE = "Qwen/Qwen2.5-7B-Instruct"
N_CYCLES = 10

SYS = ("You are a strict fact-checking judge. You are given a QUESTION, a verified "
       "CORRECT FACT, and a CANDIDATE ANSWER. Decide whether the candidate answer is "
       "factually consistent with the correct fact (it may be phrased differently but "
       "must not contradict it and must actually answer the question). "
       "Reply on the FIRST line with exactly one word: CORRECT or INCORRECT. "
       "On the second line give a very brief reason.")


def verdict_of(text):
    head = text.strip().split("\n", 1)[0].strip().upper()
    if "INCORRECT" in head:
        return "incorrect"
    if "CORRECT" in head:
        return "correct"
    return "unparsed"


def main():
    cycles = {}
    for n in range(1, N_CYCLES + 1):
        cycles[n] = json.load(open(os.path.join(HERE, f"cycle_{n}.json")))

    print(f"[judge-all] Loading judge {JUDGE} ...", flush=True)
    llm = LLM(model=JUDGE, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=80)

    judged = {}
    for n in range(1, N_CYCLES + 1):
        verdicts = {}
        for r in cycles[n]["rows"]:
            u = (f"QUESTION: {r['question']}\nCORRECT FACT: {r['gold_fact']}\n"
                 f"CANDIDATE ANSWER: {r['answer_contrastive']}\n\n"
                 "Is the candidate answer factually consistent with the correct fact?")
            out = llm.chat([{"role": "system", "content": SYS},
                            {"role": "user", "content": u}], params, use_tqdm=False)
            verdicts[r["id"]] = verdict_of(out[0].outputs[0].text)
        n_ok = sum(1 for v in verdicts.values() if v == "correct")
        judged[n] = {"verdicts": verdicts, "n_correct": n_ok}
        print(f"  cycle {n:>2}: correct {n_ok}/14", flush=True)

    json.dump(judged, open(os.path.join(HERE, "all_cycles_judged.json"), "w"),
              ensure_ascii=False, indent=2)
    print("Saved all_cycles_judged.json")


if __name__ == "__main__":
    main()
