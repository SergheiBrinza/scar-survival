#!/usr/bin/env python3
"""Judges ALL stochastic samples (10 cycles x 14 traps x K) in batch with judge Qwen2.5-7B.
For each (cycle, trap) computes the fraction of runs where the correct answer won.
Writes stoch_judged.json.
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
    cycles = {n: json.load(open(os.path.join(HERE, f"stoch_cycle_{n}.json"))) for n in range(1, N_CYCLES + 1)}

    convs, meta = [], []
    for n in range(1, N_CYCLES + 1):
        for r in cycles[n]["rows"]:
            for k, ans in enumerate(r["samples"]):
                u = (f"QUESTION: {r['question']}\nCORRECT FACT: {r['gold_fact']}\n"
                     f"CANDIDATE ANSWER: {ans}\n\n"
                     "Is the candidate answer factually consistent with the correct fact?")
                convs.append([{"role": "system", "content": SYS}, {"role": "user", "content": u}])
                meta.append((n, r["id"], k))

    print(f"[judge-stoch] {len(convs)} samples, judge {JUDGE}", flush=True)
    llm = LLM(model=JUDGE, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=80)
    outs = llm.chat(convs, params)  # batch
    verdicts = [verdict_of(o.outputs[0].text) for o in outs]

    # frac[cycle][trap] = fraction correct
    from collections import defaultdict
    cnt = defaultdict(lambda: [0, 0])  # (cycle,trap) -> [correct,total]
    for (n, tid, k), v in zip(meta, verdicts):
        cnt[(n, tid)][1] += 1
        cnt[(n, tid)][0] += (v == "correct")

    trap_ids = [r["id"] for r in cycles[1]["rows"]]
    K = cycles[1]["K"]
    per_cycle_trap = {n: {tid: round(cnt[(n, tid)][0] / cnt[(n, tid)][1], 3) for tid in trap_ids}
                      for n in range(1, N_CYCLES + 1)}
    per_trap_spread = {tid: [per_cycle_trap[n][tid] for n in range(1, N_CYCLES + 1)] for tid in trap_ids}
    # average win share of the correct answer per cycle (across all traps)
    per_cycle_winrate = [round(sum(per_cycle_trap[n][tid] for tid in trap_ids) / len(trap_ids), 3)
                         for n in range(1, N_CYCLES + 1)]

    out = {"judge": JUDGE, "n_cycles": N_CYCLES, "K": K, "trap_ids": trap_ids,
           "per_cycle_trap_winfrac": per_cycle_trap,
           "per_trap_spread": per_trap_spread,
           "per_cycle_winrate": per_cycle_winrate}
    json.dump(out, open(os.path.join(HERE, "stoch_judged.json"), "w"), ensure_ascii=False, indent=2)
    print("per_cycle_winrate:", per_cycle_winrate)
    print("Saved stoch_judged.json")


if __name__ == "__main__":
    main()
