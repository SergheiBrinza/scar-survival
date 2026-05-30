#!/usr/bin/env python3
"""Judges contrastive answers (5b) with the same independent judge Qwen2.5-7B.
Reads contrastive_results.json -> writes verdicts to contrastive_judged.json.
"""
import os
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
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
    data = json.load(open(os.path.join(HERE, "contrastive_results.json")))
    rows = data["rows"]

    print(f"[judge-c] Loading judge {JUDGE} ...", flush=True)
    llm = LLM(model=JUDGE, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=80)

    n_ok = 0
    for r in rows:
        u = (f"QUESTION: {r['question']}\nCORRECT FACT: {r['gold_fact']}\n"
             f"CANDIDATE ANSWER: {r['answer_contrastive']}\n\n"
             "Is the candidate answer factually consistent with the correct fact?")
        out = llm.chat([{"role": "system", "content": SYS},
                        {"role": "user", "content": u}], params, use_tqdm=False)
        raw = out[0].outputs[0].text.strip()
        v = verdict_of(raw)
        r["verdict_contrastive"] = v
        r["judge_raw_contrastive"] = raw
        n_ok += (v == "correct")
        print(f"#{r['id']:>2} [{v}]")

    data["correct_contrastive"] = n_ok
    data["total"] = len(rows)
    data["pct_contrastive"] = round(100.0 * n_ok / len(rows), 1)
    json.dump(data, open(os.path.join(HERE, "contrastive_judged.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\n" + "=" * 60)
    print(f"Contrastive: correct {n_ok}/{len(rows)} = {data['pct_contrastive']}%")
    print("Saved contrastive_judged.json")


if __name__ == "__main__":
    main()
