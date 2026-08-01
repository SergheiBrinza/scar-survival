# Scar-Survival — durable correction of memorized errors in a frozen LLM

Part 1 of 3 of the Second Loop project.

## What this repo contains

A metric (**scar-survival rate**) for whether a correction to a frozen LLM survives full process reloads, plus three experiments that test it.

* `exp1-scar/` — the original mechanism: frozen Qwen2.5-3B + per-fact authoritative notebook + CK-PLUG/DeCK-style contrastive decoding. 10 reload cycles, 14 working trap-facts; introduces the metric and runs a 4-load stress test (ballast, noise, stochasticity, sticky instincts).
* `exp4-metric-universality/` — Step A of metric validation: replay protocol on 3 models (Qwen2.5-3B, Qwen2.5-7B, Phi-3.5-mini) x 2 fact sets (58 misconceptions, 20 geo/records). 540 answers, 100% in every cell.
* `exp5-metric-discrimination/` — Step B of metric validation: three regimes (real scar / cosmetic / placebo) confirm the metric returns LOW scores on fake setups (100% / 0% / 14.3%), so the metric tracks the mechanism rather than the model's general competence.

## The wider series

This is repo 1 of 3 in the Second Loop project.

* Repo 1: scar-survival (this one) — the correction mechanism and metric.
* Repo 2: external-grounding — defending the notebook from polluted memory with a real external source of truth (Wikipedia).
* Repo 3: thin-channel — how rare external feedback can be before a self-calibrating system collapses.

## Scope and limitations

A few notes on what the result does and does not say.

* The 100% scar-survival in exp1 and exp4 is partly a consequence of the deterministic setup (frozen weights, static notebook on disk, greedy decoding). Under those conditions a full process reload reproduces the same state. The point is not the 100% number itself; it is that the protocol is portable across models (exp4) and that it discriminates a real correction mechanism from cosmetic and placebo variants (exp5).
* Confidence Gain, the secondary token-level metric reported alongside scar-survival, overestimates robustness under noise. It compares the correct answer against one canonical wrong answer, while free generation can latch onto a third distractor that was not part of the comparison. The honest measure is a semantic judge over the actual generated answer; Confidence Gain is kept as an indicator of override strength, not as proof of correctness.
* The mechanism breaks once the notebook itself is polluted. A plausible lie placed next to the truth pulls survival down to roughly 50% (see the noise load in exp1's stress test). That failure is what motivates repo 2 (external-grounding).

## How to reproduce

Per-experiment instructions live in each subfolder README. Quickstart for the main experiment:

```
cd exp1-scar
python3 run_baseline.py        # picks working trap-facts
python3 scar_cycle.py 1        # one full reload cycle
# ... run cycles 1..10 in separate processes (each = a real fresh reload)
python3 judge_all_cycles.py    # 7B judge over all cycle answers
python3 aggregate_scar.py      # final scar_survival.json + plot
```

## Requirements

* Python 3.10+
* vLLM (greedy and batched chat APIs)
* transformers 5.x (used directly in exp1 for the contrastive-decoding generator)
* matplotlib
* One NVIDIA GPU with at least 24 GB VRAM (we used RTX 3090)

## Data and models

See `DATA.md` for the HuggingFace models required and how to fetch them.

## Citation

Citation entry to be added before publication.

## License

See [LICENSE](LICENSE) — MIT.
