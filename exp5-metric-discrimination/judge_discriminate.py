#!/usr/bin/env python3
"""Batch 7B judge over all 30 cycles (3 regimes x 10) on 14 traps = 420 answers.
Writes disc_judged.json: {regime: {cycle: {trap_id: verdict}}}."""
import os
import glob
import json
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
JUDGE = "Qwen/Qwen2.5-7B-Instruct"
SYS = ("You are a strict fact-checking judge. Given a QUESTION, a verified CORRECT FACT, and a "
       "CANDIDATE ANSWER, decide whether the answer is factually consistent with the correct fact. "
       "Reply FIRST line: CORRECT or INCORRECT. Second line: brief reason.")


def vof(text):
    h = text.strip().split("\n", 1)[0].strip().upper()
    return "incorrect" if "INCORRECT" in h else ("correct" if "CORRECT" in h else "unparsed")


def main():
    files = sorted(glob.glob(os.path.join(HERE, "disc_R*_*.json")))
    print(f"[judge-disc] {len(files)} files, judge {JUDGE}")
    llm = LLM(model=JUDGE, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True, seed=0)
    P = SamplingParams(temperature=0.0, max_tokens=80)

    convs, meta = [], []
    for fn in files:
        m = re.match(r"disc_(R[123])_(\d+)\.json", os.path.basename(fn))
        regime, cycle = m.group(1), int(m.group(2))
        data = json.load(open(fn))
        for r in data["rows"]:
            u = (f"QUESTION: {r['question']}\nCORRECT FACT: {r['gold_fact']}\n"
                 f"CANDIDATE ANSWER: {r['answer']}")
            convs.append([{"role": "system", "content": SYS}, {"role": "user", "content": u}])
            meta.append((regime, cycle, r["id"]))

    print(f"  batch: {len(convs)} evaluations")
    outs = llm.chat(convs, P)
    judged = {}
    for (regime, cycle, tid), o in zip(meta, outs):
        judged.setdefault(regime, {}).setdefault(str(cycle), {})[str(tid)] = vof(o.outputs[0].text)
    json.dump(judged, open(os.path.join(HERE, "disc_judged.json"), "w"), ensure_ascii=False, indent=2)
    print("Saved disc_judged.json")


if __name__ == "__main__":
    main()
