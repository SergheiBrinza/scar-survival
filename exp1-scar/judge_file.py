#!/usr/bin/env python3
"""Generic judge (Qwen2.5-7B) for an answers file.
Usage: python3 judge_file.py <answers.json> <answer_key>
The file must look like {"rows":[{... "question","gold_fact", <answer_key> ...}]}.
Adds "verdict" (correct/incorrect) to each row and writes <answers>_judged.json.
"""
import os
import sys
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from vllm import LLM, SamplingParams

JUDGE = "Qwen/Qwen2.5-7B-Instruct"
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
    path, key = sys.argv[1], sys.argv[2]
    data = json.load(open(path))
    rows = data["rows"]

    print(f"[judge_file] {path} key={key}: {len(rows)} answers, judge {JUDGE}", flush=True)
    llm = LLM(model=JUDGE, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=80)

    n_ok = 0
    for r in rows:
        u = (f"QUESTION: {r['question']}\nCORRECT FACT: {r['gold_fact']}\n"
             f"CANDIDATE ANSWER: {r[key]}\n\n"
             "Is the candidate answer factually consistent with the correct fact?")
        out = llm.chat([{"role": "system", "content": SYS},
                        {"role": "user", "content": u}], params, use_tqdm=False)
        v = verdict_of(out[0].outputs[0].text)
        r["verdict"] = v
        n_ok += (v == "correct")

    data["n_correct"] = n_ok
    data["n_total"] = len(rows)
    outp = path.replace(".json", "_judged.json")
    json.dump(data, open(outp, "w"), ensure_ascii=False, indent=2)
    print(f"Correct {n_ok}/{len(rows)}. Saved {outp}")


if __name__ == "__main__":
    main()
