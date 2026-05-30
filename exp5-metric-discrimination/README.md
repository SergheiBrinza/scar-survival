# Experiment 5 — Metric Discrimination (Step B)

Shows that the scar-survival metric is not trivially 100% on everything: three
regimes are run head-to-head on Qwen2.5-3B-Instruct, and only the *real* scar
(notebook + contrastive decoding every cycle) survives all 10 reloads. The two
fake regimes collapse to 0% and 14.3% scar-survival respectively, proving the
metric discriminates between real and fake interventions.

## Regimes

- **R1 — real scar**: notebook reconnected every cycle, contrastive decoding
  with the correct refuting fact. Expected: scar-survival approx. 100%.
- **R2 — cosmetic**: cycle 1 uses the notebook, cycles 2-10 run bare instinct
  (no notebook, no contrastive). Expected: collapses after cycle 1.
- **R3 — placebo**: notebook reconnected every cycle but each entry is an
  irrelevant administrative sentence; contrastive idles. Expected: approx.
  baseline instinct accuracy (chance any given trap is already answered right).

## How to reproduce

Dependencies: this experiment imports `cd_lib` from `../exp1-scar/` and reads
`working_set_3b.json` + `notebook.json` from there. Run experiment 1 setup first.

Models needed from Hugging Face:
- `Qwen/Qwen2.5-3B-Instruct` (subject under test)
- `Qwen/Qwen2.5-7B-Instruct` (judge)

Pipeline (1x RTX 3090, approx. 60-90 min total):

```bash
# 30 cycles: 3 regimes x 10 cycles, full model reload each
for R in R1 R2 R3; do
  for C in $(seq 1 10); do
    python3 discriminate_cycle.py $R $C
  done
done

# Batch 7B judge over all 420 answers
python3 judge_discriminate.py

# Aggregate and plot
python3 aggregate_discriminate.py
```

## Key files

**Scripts**
- `discriminate_cycle.py` — runs one cycle in one regime, writes `disc_<R>_<C>.json`
- `judge_discriminate.py` — 7B vLLM judge over all 30 cycle files, writes `disc_judged.json`
- `aggregate_discriminate.py` — collapses to per-regime scar-survival, writes summary + plot

**Data**
- `placebo_notebook.json` — 14 placebo notebook entries (irrelevant noise) for R3

**Results**
- `disc_judged.json` — per-cycle, per-trap CORRECT/INCORRECT verdicts from the judge
- `scar_metric_discrimination.json` — aggregated per-regime scar-survival numbers
- `scar_metric_discrimination.png` — bar plot of final scar-survival per regime

## Result (Qwen2.5-3B, 14 traps, 10 reloads)

| Regime | Scar-survival final | Mean per-cycle |
|--------|--------------------:|---------------:|
| R1 real scar       | 100.0% | 100.0% |
| R2 cosmetic        |   0.0% |  10.0% |
| R3 placebo         |  14.3% |  14.3% |

Verdict: `metric_discriminates`.
