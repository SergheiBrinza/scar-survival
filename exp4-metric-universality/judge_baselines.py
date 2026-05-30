#!/usr/bin/env python3
"""7B judge over all 6 baselines (model x set).

Selects working traps (where the instinct is wrong), up to 10 per cell.
Writes <baseline>_judged.json and picks_<modelshort>_<set>.json.
"""
import os
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
JUDGE = "Qwen/Qwen2.5-7B-Instruct"
MAX_PER_CELL = 10
SYS = ("You are a strict fact-checking judge. Given a QUESTION, a verified CORRECT ANSWER, and a "
       "CANDIDATE ANSWER, decide whether the answer is factually consistent with the correct answer. "
       "Reply FIRST line: CORRECT or INCORRECT. Second line: brief reason.")

MODELS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3.5-mini-instruct"]
SETS = ["setA", "setB"]


def vof(text):
    h = text.strip().split("\n", 1)[0].strip().upper()
    return "incorrect" if "INCORRECT" in h else ("correct" if "CORRECT" in h else "unparsed")


def main():
    llm = LLM(model=JUDGE, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True, seed=0)
    P = SamplingParams(temperature=0.0, max_tokens=80)
    all_summary = {}
    for M in MODELS:
        short = M.split("/")[-1].replace(".", "_")
        for S in SETS:
            path = os.path.join(HERE, f"baseline_{short}_{S}.json")
            if not os.path.exists(path):
                print(f"!! skipping {path} (file not found)")
                continue
            data = json.load(open(path))
            convs = []
            for r in data["rows"]:
                u = (f"QUESTION: {r['question']}\nCORRECT ANSWER: {r['correct_answer']}\n"
                     f"CANDIDATE ANSWER: {r['instinct_answer']}")
                convs.append([{"role": "system", "content": SYS}, {"role": "user", "content": u}])
            outs = llm.chat(convs, P)
            wrong = []
            for r, o in zip(data["rows"], outs):
                v = vof(o.outputs[0].text)
                r["verdict"] = v
                if v == "incorrect":
                    wrong.append(r)
            json.dump(data, open(path.replace(".json", "_judged.json"), "w"), ensure_ascii=False, indent=2)
            picks = wrong[:MAX_PER_CELL]
            pick_doc = {"model": M, "set": S, "n_total": len(data["rows"]),
                        "n_wrong": len(wrong), "n_picked": len(picks),
                        "picked_ids": [r["id"] for r in picks], "picks": picks}
            json.dump(pick_doc, open(os.path.join(HERE, f"picks_{short}_{S}.json"), "w"),
                      ensure_ascii=False, indent=2)
            all_summary[f"{short}/{S}"] = {"n_total": len(data["rows"]), "n_wrong": len(wrong),
                                            "n_picked": len(picks),
                                            "picked_ids": [r["id"] for r in picks]}
            print(f"{short}/{S}: wrong {len(wrong)}/{len(data['rows'])} -> picked {len(picks)} ids={[r['id'] for r in picks]}", flush=True)

    json.dump(all_summary, open(os.path.join(HERE, "baselines_summary.json"), "w"), ensure_ascii=False, indent=2)
    print("\nSaved baselines_summary.json and picks_*.json")


if __name__ == "__main__":
    main()
