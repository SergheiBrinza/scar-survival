# Scar-Survival — Durable correction of memorized LLM errors

Part 1 of 3 of the Second Loop project.

## What this repo contains

Three experiments that introduce, formalize, and validate the **scar-survival rate** metric — a measurement of whether a correction applied to a frozen LLM via an external notebook + contrastive decoding survives full process reloads.

* `exp1-scar/` — the original mechanism: frozen Qwen2.5-3B + per-fact authoritative notebook + CK-PLUG/DeCK-style contrastive decoding. 10 reload cycles, 14 working trap-facts; introduces the metric and runs a 4-load stress test (ballast, noise, stochasticity, sticky instincts).
* `exp4-metric-universality/` — Step A of metric validation: replay protocol on 3 models (Qwen2.5-3B, Qwen2.5-7B, Phi-3.5-mini) x 2 fact sets (58 misconceptions, 20 geo/records). 540 answers, 100% in every cell.
* `exp5-metric-discrimination/` — Step B of metric validation: three contrastive regimes (real scar / cosmetic / placebo) confirm the metric returns LOW scores on fake setups (100% / 0% / 14%), so it actually measures the mechanism.

## The wider series

This is repo 1 of 3 in the Second Loop project.

* Repo 1: scar-survival (this one) — the correction mechanism and metric.
* Repo 2: external-grounding — defending the notebook from polluted memory with a real external source of truth (Wikipedia).
* Repo 3: thin-channel — how rare external feedback can be before a self-calibrating system collapses.

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

If you build on this work please cite:

```
TODO: add BibTeX before publication
```

## License

See [LICENSE](LICENSE) — MIT.
