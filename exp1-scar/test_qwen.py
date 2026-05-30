#!/usr/bin/env python3
"""Short sanity check for Qwen2.5-7B-Instruct via vLLM on a single RTX 3090 (GPU 0)."""
import os

# Use ONLY one GPU (the zeroth)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    print(f"[test] Loading model {MODEL} on GPU 0 ...", flush=True)
    llm = LLM(
        model=MODEL,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enforce_eager=True,  # faster startup, no long graph compilation
    )
    print("[test] Model loaded. Asking a question.", flush=True)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2 and why is the sky blue?"},
    ]

    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=400)
    outputs = llm.chat(messages, params)

    answer = outputs[0].outputs[0].text.strip()
    print("\n" + "=" * 60)
    print("QUESTION: What is 2+2 and why is the sky blue?")
    print("=" * 60)
    print("MODEL ANSWER:\n")
    print(answer)
    print("=" * 60)


# Guard for multiprocessing spawn (vLLM requirement)
if __name__ == "__main__":
    main()
