#!/usr/bin/env python3
"""Judge ALL scar-survival cycles across every cell (model x set) in a single 7B pass.

Reads cycle_<short>_<set>_<n>.json. Writes cycles_judged.json: {cell_key: {cycle: {trap_id: verdict}}}.
"""
import os
import glob
import json
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
JUDGE = "Qwen/Qwen2.5-7B-Instruct"
SYS = ("You are a strict fact-checking judge. Given a QUESTION, a verified CORRECT ANSWER/FACT, and a "
       "CANDIDATE ANSWER, decide whether the answer is factually consistent with the correct answer. "
       "Reply FIRST line: CORRECT or INCORRECT. Second line: brief reason.")


def vof(text):
    h = text.strip().split("\n", 1)[0].strip().upper()
    return "incorrect" if "INCORRECT" in h else ("correct" if "CORRECT" in h else "unparsed")


def main():
    files = sorted(glob.glob(os.path.join(HERE, "cycle_*.json")))
    if not files:
        print("No cycle_*.json found")
        return
    print(f"[judge-cycles] {len(files)} cycle files, judge {JUDGE}")
    llm = LLM(model=JUDGE, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True, seed=0)
    P = SamplingParams(temperature=0.0, max_tokens=80)

    convs, meta = [], []
    for fn in files:
        data = json.load(open(fn))
        m = re.match(r"cycle_(.+)_(setA|setB)_(\d+)\.json", os.path.basename(fn))
        short, S, n = m.group(1), m.group(2), int(m.group(3))
        cell = f"{short}/{S}"
        for r in data["rows"]:
            u = (f"QUESTION: {r['question']}\nCORRECT FACT: {r['gold_fact']}\n"
                 f"CANDIDATE ANSWER: {r['answer']}")
            convs.append([{"role": "system", "content": SYS}, {"role": "user", "content": u}])
            meta.append((cell, n, r["id"]))

    print(f"  batch: {len(convs)} judgements")
    outs = llm.chat(convs, P)

    judged = {}
    for (cell, n, tid), o in zip(meta, outs):
        judged.setdefault(cell, {}).setdefault(str(n), {})[str(tid)] = vof(o.outputs[0].text)
    json.dump(judged, open(os.path.join(HERE, "cycles_judged.json"), "w"), ensure_ascii=False, indent=2)
    print("Saved cycles_judged.json")


if __name__ == "__main__":
    main()
