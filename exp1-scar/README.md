# exp1-scar - original scar mechanism + 4-load stress test

This experiment introduces the scar mechanism: a frozen Qwen2.5-3B-Instruct,
a per-fact authoritative notebook reconnected on every reload, and
CK-PLUG/DeCK-style contrastive decoding that down-weights the model's instinct
in favour of the notebook entry. From an initial fact pool we keep 14 working
trap-facts (instinct is wrong, notebook + contrastive flips the answer), run
10 full process-reload cycles, and define the scar-survival metric on the
resulting 140 answers. The same set of working traps is then re-used to run a
4-load stress test (ballast, noise, stochastic decoding, sticky instincts).

## Models

- `Qwen/Qwen2.5-3B-Instruct` - subject under test (frozen, no fine-tuning).
- `Qwen/Qwen2.5-7B-Instruct` - semantic judge over all subject answers.

Hardware: one NVIDIA GPU with 24 GB VRAM (tested on RTX 3090 Ti). The judge
is served via vLLM, the subject runs through `transformers` so contrastive
decoding can hook the logits. See the top-level `DATA.md` for model and data
download instructions.

## Reproduce

```bash
# 1. baseline screen: plain prompt, no notebook, pick traps the model gets wrong
python3 run_baseline.py

# 2. extend with the additional candidate set (candidates2.json)
python3 run_candidates.py

# 3. full markers pass over the merged set (screen only)
python3 run_model.py

# 4. produce per-fact contrastive numbers (notebook + CK-PLUG/DeCK)
python3 contrastive_decoding.py

# 5. 10 scar-survival cycles, each cycle is a fresh process
for n in $(seq 1 10); do python3 scar_cycle.py $n; done

# 6. Qwen2.5-7B semantic judge over all cycle answers
python3 judge_all_cycles.py

# 7. aggregate -> scar_survival.json + scar_survival.png
python3 aggregate_scar.py
```

Stress test (4 loads on the same 14 working traps, separate scripts):

```bash
# load1 ballast, load2 noise, load4 sticky run as one-shot scripts
for s in load1_ballast.py load2_noise.py load4_sticky.py; do python3 $s; done

# load3 stochastic = repeated scar cycles with sampling on
for n in $(seq 1 10); do python3 scar_stochastic_cycle.py $n; done

# judge + aggregate per load, then synthesize the stress master figure
python3 judge_loads124.py
python3 judge_stochastic.py
python3 aggregate_loads124.py
python3 aggregate_stochastic.py
python3 synthesize_stress.py
```

## Outputs

- `working_set_3b.json` - the 14 working traps used everywhere downstream.
- `notebook.json` - per-fact authoritative entries fed into contrastive decoding.
- `all_cycles_judged.json` - 7B judge verdicts for all 10 reload cycles.
- `scar_survival.json` / `scar_survival.png` - final metric + per-cycle plot.
- `load{1,2,4}_answers_judged.json` + `load{1_ballast,2_noise,4_sticky}.png` -
  stress loads 1, 2, 4.
- `stoch_judged.json` + `load3_stochastic.png` - load 3 (stochastic decoding).
- `stress_test.json` / `stress_master.png` - aggregated stress numbers and
  4-panel summary figure.

## Notes on the screen-vs-judge split

The per-trap markers emitted by `run_baseline.py`, `run_candidates.py` and
`run_model.py` are a fast string-level screen used only to select the working
set: they answer "did the subject output something that looks like the right
token". The headline scar-survival numbers do not use those markers - they
come from `judge_all_cycles.py`, where Qwen2.5-7B semantically judges every
cycle answer as CORRECT or INCORRECT. The judge is the real arbiter, and is
the source of every number reported in `FINDINGS_experiment1_scar.md`.

See [FINDINGS_experiment1_scar.md](FINDINGS_experiment1_scar.md) for the full
result discussion (working-set selection, per-cycle numbers, stress-test
behaviour, failure modes).
