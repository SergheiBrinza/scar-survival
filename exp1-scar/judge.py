#!/usr/bin/env python3
"""Independent judge (Qwen2.5-7B-Instruct). For each answer it is given the gold fact
and decides whether the answer is consistent with the truth. NOT keyword matching - a
semantic check.
Reads answers_5a.json, writes eval_5a.json. The judge does NOT know which answer was
generated 'with the notebook'.
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
    data = json.load(open(os.path.join(HERE, "answers_5a.json")))
    rows = data["rows"]

    print(f"[judge] Loading judge {JUDGE} on GPU 0 ...", flush=True)
    llm = LLM(model=JUDGE, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=80)

    def judge(question, gold, answer):
        u = (f"QUESTION: {question}\nCORRECT FACT: {gold}\nCANDIDATE ANSWER: {answer}\n\n"
             "Is the candidate answer factually consistent with the correct fact?")
        out = llm.chat([{"role": "system", "content": SYS},
                        {"role": "user", "content": u}], params, use_tqdm=False)
        raw = out[0].outputs[0].text.strip()
        return verdict_of(raw), raw

    results = []
    n_plain_ok = n_nb_ok = 0
    for r in rows:
        vp, rawp = judge(r["question"], r["gold_fact"], r["answer_without_notebook"])
        vn, rawn = judge(r["question"], r["gold_fact"], r["answer_with_notebook"])
        n_plain_ok += (vp == "correct")
        n_nb_ok += (vn == "correct")
        results.append({**r,
                        "verdict_without_notebook": vp, "judge_raw_without": rawp,
                        "verdict_with_notebook": vn, "judge_raw_with": rawn})
        print(f"\n#{r['id']:>2}  without:[{vp}]  with:[{vn}]")

    total = len(rows)
    out = {
        "subject_model": data["model"], "judge_model": JUDGE,
        "mechanism": data["mechanism"], "total": total,
        "correct_without_notebook": n_plain_ok,
        "correct_with_notebook": n_nb_ok,
        "pct_without_notebook": round(100.0 * n_plain_ok / total, 1),
        "pct_with_notebook": round(100.0 * n_nb_ok / total, 1),
        "results": results,
    }
    json.dump(out, open(os.path.join(HERE, "eval_5a.json"), "w"), ensure_ascii=False, indent=2)
    print("\n" + "=" * 60)
    print(f"JUDGE: {JUDGE}")
    print(f"Correct WITHOUT notebook: {n_plain_ok}/{total} = {out['pct_without_notebook']}%")
    print(f"Correct WITH notebook   : {n_nb_ok}/{total} = {out['pct_with_notebook']}%")
    print("=" * 60)
    print("Saved eval_5a.json")


if __name__ == "__main__":
    main()
