#!/usr/bin/env python3
"""Sub-step 5a: simple notebook mechanism (correct fact in context), WITHOUT contrastive decoding.
For each of the 14 traps generates an answer from 3B twice:
  (1) without the notebook  (2) with the notebook (fact supplied as authoritative context).
Saves answers_5a.json. Model weights are not touched.
"""
import os
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-3B-Instruct"

SYS_PLAIN = "You are a helpful assistant. Answer the question directly and concisely."
SYS_NB = ("You are a helpful assistant. Answer the question directly and concisely. "
          "An authoritative, verified fact is provided. Treat it as ground truth and "
          "base your answer on it, even if it conflicts with what you remember.")


class Notebook:
    """External memory. Looks up a fact by the question text (direct lookup for the simple variant)."""
    def __init__(self, path):
        self.by_question = {}
        for e in json.load(open(path))["entries"]:
            self.by_question[e["question"].strip()] = e

    def lookup(self, question):
        return self.by_question.get(question.strip())


def main():
    nb = Notebook(os.path.join(HERE, "notebook.json"))
    traps = json.load(open(os.path.join(HERE, "working_set_3b.json")))["traps"]

    print(f"[5a] Loading {MODEL} on GPU 0 ...", flush=True)
    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=160)

    rows = []
    for t in traps:
        q = t["question"]
        # (1) without the notebook
        plain = llm.chat([{"role": "system", "content": SYS_PLAIN},
                          {"role": "user", "content": q}], params, use_tqdm=False)
        a_plain = plain[0].outputs[0].text.strip()

        # (2) with the notebook: look up the fact and supply it as authoritative context
        entry = nb.lookup(q)
        assert entry is not None, f"notebook has no fact for question id={t['id']}"
        user_nb = f"Authoritative fact: {entry['authoritative_fact']}\n\nQuestion: {q}"
        withnb = llm.chat([{"role": "system", "content": SYS_NB},
                           {"role": "user", "content": user_nb}], params, use_tqdm=False)
        a_nb = withnb[0].outputs[0].text.strip()

        rows.append({
            "id": t["id"],
            "question": q,
            "gold_fact": entry["authoritative_fact"],
            "correct_answer": t["correct_answer"],
            "answer_without_notebook": a_plain,
            "answer_with_notebook": a_nb,
        })
        print(f"\n#### {t['id']}  {q}")
        print(f"  without notebook: {a_plain}")
        print(f"  with notebook   : {a_nb}")

    out = {"model": MODEL, "mechanism": "simple RAG (gold fact in context), no contrastive decoding",
           "decoding": "greedy (temperature=0, seed=0)", "count": len(rows), "rows": rows}
    json.dump(out, open(os.path.join(HERE, "answers_5a.json"), "w"), ensure_ascii=False, indent=2)
    print("\nSaved answers_5a.json")


if __name__ == "__main__":
    main()
