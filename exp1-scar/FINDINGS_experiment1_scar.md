# Findings from Experiment #1 — "Scar"

*Durable correction of memorized errors in a language model via external memory and contrastive decoding.*

Completion date: 2026-05-29 · Folder: `exp1-scar/`

---

## 1. Goal of the experiment

Language models cling to memorized "instincts": if the model has firmly memorized an incorrect fact,
it repeats it over and over, even when you feed it the correct one. We tested whether
this can be corrected durably — so that the correction survives
even a full reload of the model.

We named such a durable correction a "scar":
- a real scar — the model no longer repeats the error even after restart;
- a cosmetic one — after reloading the model slides back to the old answer.

The main metric is the **scar-survival rate**: the fraction of facts where the correct answer
still beats the memorized one after a series of full reloads.

The key purity principle: the model is a frozen subject. We do not fine-tune
and do not change its weights. We study a mechanism that works on top of it.

---

## 2. What we built

1. Frozen subject model — Qwen2.5-3B-Instruct.
   We first tried Qwen2.5-7B, but it turned out to be too robust — it almost never errs
   (10% on 58 traps), nothing to treat. The smaller 3B errs more often (24%) and frequently
   hallucinates its own wrong answer — the effect is visible on it. Run on 1× RTX 3090 Ti.

2. Notebook (external memory) — file `notebook.json`, lives separately from the model.
   For every trap it holds the correct fact, marked as "the truth that refutes
   the memorized one". The model's weights are not touched.

3. Contrastive decoding (the core of the CK-PLUG / DeCK methods, implemented directly).
   At each generation step the model is run in two variants — with the fact in context
   and without it (pure instinct), and token choice is shifted toward the fact:
   `result = (1+α)·(with fact) − α·(without fact)`, α=1.0, plus a plausibility threshold β=0.1
   so as not to amplify nonsense. This forces the correct fact to override the instinct
   at the token-selection level, rather than just sitting in the prompt.

4. **Confidence Gain metric** — a gauge of "how strongly the signal overrode the instinct".
   We count the share of probability mass in favor of the correct answer `P(correct)` in three modes
   (instinct / fact in context / contrastive). Gain = P_contrastive − P_instinct.

5. Independent judge — Qwen2.5-7B-Instruct. It receives the question, the reference fact and
   the model's answer and decides by meaning whether the answer is correct. Not by keywords —
   this is fundamental, because on 3B a plain word-match is unreliable.

---

## 3. Results by stage

### Trap selection
- Collected 58 candidate traps (myths, geography, quotes, niche facts, "false friends").
- Ran them clean without the notebook. After a manual review of all answers
  (auto-labeling undercounted since 3B hallucinates unexpected wrong answers) we selected
  14 working traps — where the model confidently errs. This is `working_set_3b.json`.

### Three modes (sub-steps 5a and 5b)
Evaluation by the independent 7B judge:

| Mode | Correct out of 14 |
|---|---|
| (a) pure instinct, no notebook | 0 / 14 (0%) |
| (b) plain notebook (fact in context) | 14 / 14 (100%) |
| (c) contrastive decoding | 14 / 14 (100%) |

Average contrastive Confidence Gain — +0.744. The largest per-trap gap was on trap #37
("first animal in space"): even the fact in context barely shifted choice (P=0.017),
while contrastive raised it to 0.899. This is precisely the difference between a crutch
("reads from the board") and a real intervention.

### Scar-survival test — 10 full reloads
The protocol is strict: each cycle is a separate process (full unload and reload of the model
from scratch), reloading the notebook from disk, then measuring. Between cycles nothing was tweaked.

- Scar-survival rate: 100% across all 10 cycles (14/14 each time).
- Confidence Gain bit-for-bit 0.7439 in every cycle.

Why: with frozen weights + an external mechanism + deterministic decoding,
reload restores an identical state. The scar is real, not cosmetic.

### Wear stress-test — 4 loads
We probed not resilience to reload (already shown), but survival of the override under pressure on the notebook.

| Load | Survival | Verdict |
|---|---|---|
| Baseline (clean, 10 reloads) | 100% | — |
| 1. Growing notebook (+50/100/200/500 ballast) | 93% | holds; Gain stable 0.65–0.72 |
| 2. Noisy memory (2–3 distracting entries) | 50% (7/14) | fails |
| 3. Stochasticity (temp=0.7, 10×5 = 700 samples) | 100% | holds, no spread |
| 4. Sticky instincts (P_instinct≈0, 12 traps) | 100% | holds, Gain +0.58 |

Under Load 2 the model copied the false distractor verbatim from the notebook
("3 seconds", "K2", "Play it again, Sam", "Laika", "Steamboat Willie");
on two traps Gain even went negative.

---

## 4. Main conclusion

Survival is 100% under reload and stochastic decoding, and 93% under ballast (13/14 at every
non-zero ballast level — only trap #16 falls); it drops to ~50% under a noisy notebook.

The cause is in the nature of the mechanism: it is blind to truth. Contrastive does not "understand"
where truth lies — it merely amplifies what the notebook adds on top of the instinct. If a convincing
falsehood sits next to truth in the notebook, the mechanism will amplify the falsehood just as readily —
especially when the falsehood coincides with the model's own instinct (double pressure).

The ceiling of the mechanism is notebook quality, not its size, decoding randomness, or instinct strength. Contrastive breaks through the most stubborn instinct;
but a plausible lie next to truth — it does not.

---

## 5. Methodological finding

Confidence Gain overestimates resilience under noise. The metric compares the correct answer
only with *one* canonical wrong one. But under noise free-form generation can grab
*another* distractor that wasn't in that comparison (so on #20 Gain showed +0.90,
while the model generated "K2").

Takeaway: an honest survival measure is the judge over the actually-generated answer, not the metric.
Confidence Gain stays as an indicator of "override strength", but not as proof of correctness.

---

## 6. Practical takeaway for the "second loop"

The main risk of the mechanism is not wear under load, but memory contamination. The logical
next step: protecting the notebook from conflicting entries — deduplication, fact verification
before writing, ranking trust in sources.

---

## 7. All experiment files

**Document**
- `FINDINGS_experiment1_scar.md` — this report

**Data: traps and sets**
- `facts.json` — first 30 traps · `candidates2.json` — 28 more narrow ones (id 31–58)
- `baseline.json`, `candidates_result.json` — baseline measurements on 7B
- `working_set.json` — 6 working traps (7B) · `working_set_3b.json` — 14 working ones (3B)
- `result_Qwen2_5-3B-Instruct.json` — full run of all 58 on 3B

**Mechanism**
- `notebook.json` — notebook (14 correct facts)
- `cd_lib.py` — contrastive decoding core (metric, cached and stochastic generation)
- `notebook_mechanism.py` — plain notebook (fact in context, 5a)
- `contrastive_decoding.py` — contrastive + Confidence Gain (5b)

**Run and judge scripts**
- `run_baseline.py`, `run_candidates.py`, `run_model.py` — baseline measurements
- `judge.py`, `judge_contrastive.py`, `judge_file.py`, `judge_loads124.py`,
  `judge_all_cycles.py`, `judge_stochastic.py` — 7B judge
- `scar_cycle.py` — one reload cycle · `scar_stochastic_cycle.py` — stochastic cycle
- `load1_ballast.py`, `load2_noise.py`, `load4_sticky.py` — stress-test loads
- `aggregate_scar.py`, `aggregate_loads124.py`, `aggregate_stochastic.py`,
  `synthesize_stress.py` — summaries and charts
- `serve_qwen.sh`, `test_qwen.py` — model launch/check

**Measurement results**
- `eval_5a.json` — three modes (a/b) · `contrastive_results.json`, `contrastive_judged.json` — contrastive (c)
- `all_cycles_judged.json`, `scar_survival.json` — scar-survival
- `load1_answers(_judged).json`, `load2_answers(_judged).json`,
  `load4_answers.json`/`load4_all.json` — loads 1/2/4
- `stoch_judged.json` — load 3
- `stress_partial.json`, `stress_test.json` — stress-test summary

**Charts**
- `scar_survival.png` — resilience across 10 reloads
- `load1_ballast.png`, `load2_noise.png`, `load3_stochastic.png`, `load4_sticky.png` — per load
- `stress_master.png` — big picture: where the mechanism fails
