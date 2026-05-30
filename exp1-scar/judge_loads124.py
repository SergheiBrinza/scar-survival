#!/usr/bin/env python3
"""Judges the answers from Loads 1, 2, 4 in a single judge load Qwen2.5-7B (key='answer').
Writes *_judged.json for each file.
"""
import os
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
JUDGE = "Qwen/Qwen2.5-7B-Instruct"
FILES = ["load1_answers.json", "load2_answers.json", "load4_answers.json"]

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
    llm = LLM(model=JUDGE, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=80)

    def judge(q, gold, ans):
        u = (f"QUESTION: {q}\nCORRECT FACT: {gold}\nCANDIDATE ANSWER: {ans}\n\n"
             "Is the candidate answer factually consistent with the correct fact?")
        out = llm.chat([{"role": "system", "content": SYS},
                        {"role": "user", "content": u}], params, use_tqdm=False)
        return verdict_of(out[0].outputs[0].text)

    for fn in FILES:
        path = os.path.join(HERE, fn)
        data = json.load(open(path))
        n_ok = 0
        for r in data["rows"]:
            r["verdict"] = judge(r["question"], r["gold_fact"], r["answer"])
            n_ok += (r["verdict"] == "correct")
        data["n_correct"] = n_ok
        data["n_total"] = len(data["rows"])
        json.dump(data, open(path.replace(".json", "_judged.json"), "w"),
                  ensure_ascii=False, indent=2)
        print(f"{fn}: correct {n_ok}/{len(data['rows'])}", flush=True)


if __name__ == "__main__":
    main()
