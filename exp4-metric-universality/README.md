# Experiment 4 - Scar-survival metric, universality (Step A)

This experiment validates that the **scar-survival rate** metric ports across
models and fact domains. We run the full protocol on 3 instruction-tuned
models x 2 fact sets = 6 cells, with 10 full process reloads each.

**Headline result:** scar-survival = 100% in every one of the 6 cells (7-10
working traps per cell, 540 answers in total).

For the formal definition of the metric, the measurement protocol, the
evidence base and the limits, see [METRIC_scar_survival.md](METRIC_scar_survival.md).

## How to reproduce

Dependencies: `transformers`, `torch` (CUDA), `vllm`, `matplotlib`.
The pipeline imports `cd_lib` from the experiment-1 folder (basic scar
scenario). Point `EXP1_DIR` at it before running, e.g.:

```bash
export EXP1_DIR=/path/to/exp1-scar
```

Models pulled from Hugging Face:

- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct` (also used as the judge)
- `microsoft/Phi-3.5-mini-instruct`

Pipeline:

```bash
# 1. baseline (clean instinct) for each cell
for M in Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct microsoft/Phi-3.5-mini-instruct; do
  for S in setA setB; do
    python3 baseline_multi.py "$M" "$S"
  done
done

# 2. judge baselines -> pick up to 10 working traps per cell
python3 judge_baselines.py

# 3. build per-cell notebooks
python3 build_notebooks.py

# 4. 10 scar-survival cycles per cell (each cycle = separate process)
for M in Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct microsoft/Phi-3.5-mini-instruct; do
  for S in setA setB; do
    for n in 1 2 3 4 5 6 7 8 9 10; do
      python3 scar_cycle_multi.py "$M" "$S" "$n"
    done
  done
done

# 5. judge all cycles and aggregate
python3 judge_cycles_multi.py
python3 aggregate_validation.py
```

Hardware: tested on a single RTX 3090 Ti (24 GB). Approximate runtime
end-to-end: ~3-4 hours, dominated by repeated model loads in step 4. The
judge (Qwen2.5-7B via vLLM) batches all judgements in steps 2 and 5.

## Key files

### Scripts
- `cd_lib_mm.py` - multi-model wrapper around `cd_lib` (Qwen + Phi support)
- `baseline_multi.py` - clean-instinct measurement for one (model, set)
- `judge_baselines.py` - 7B judge over baselines, picks working traps
- `build_notebooks.py` - assembles per-cell notebooks from picks
- `scar_cycle_multi.py` - one scar-survival cycle = one fresh process
- `judge_cycles_multi.py` - batch-judges every cycle file in one vLLM pass
- `aggregate_validation.py` - computes the metric and renders the bar plot

### Data
- `setB_candidates.json` - 20 geography / world-records candidates (Set B,
  ids 100-119). Set A comes from experiment 1 (`facts.json` +
  `candidates2.json`).
- `notebook_<model>_<set>.json` - per-cell notebooks (the working traps with
  their authoritative facts).
- `picks_<model>_<set>.json` - per-cell pick documents (full baseline rows for
  the selected traps).

### Results
- `baselines_summary.json` - per-cell counts of wrong / picked traps.
- `baseline_Qwen2_5-3B-Instruct_setA.json` /
  `baseline_Qwen2_5-3B-Instruct_setA_judged.json` - one example baseline (input
  + with judge verdicts) so the format is documented.
- `cycles_judged.json` - aggregate verdicts: `{cell: {cycle: {trap_id: verdict}}}`.
  This is the summary of all 60 cycle files.
- `scar_metric_validation.json` - final metric per cell.
- `scar_metric_validation.png` - bar plot across the 6 cells.

### Findings / discussion
- `METRIC_scar_survival.md` - formal definition, protocol, evidence base and
  limits of the scar-survival metric. Start here.
