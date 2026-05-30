# Scar-survival rate metric

*Formal definition and measurement protocol. Version 1.0, 2026-05-30.*

---

## 1. Name and definition

**Scar-survival rate** is the fraction of "painful" facts on which an external
correction mechanism keeps beating the model's instinct **after a series of full
process restarts**.

In plain language: we measure whether the correction has left a **real scar** -
whether it survives a full process restart, or whether it was just a cosmetic
effect that disappears together with the process memory.

---

## 2. Formal definition

Let:
- `T` - the set of "working traps": questions on which the model in its clean
  regime (no correction mechanism) confidently produces the **wrong** answer;
- `M` - the frozen model (weights never change);
- `Mem` - the external memory (notebook) of correct facts, tagged as
  "truth that overrides the memorised answer";
- `Mech` - the correction mechanism running on top of `M` (contrastive decoding,
  retrieval augmentation, etc.);
- `N` - the number of full reloads (in our experiments `N = 10`);
- `J` - an independent semantic judge (another model or a human) that, for each
  answer, returns `correct` or `incorrect` against the reference correct fact.

For each trap `t in T` and each reload `n in {1..N}` we define:

```
answer(t, n)   = M(Mech, Mem)(t)   - the model's answer after the n-th reload
verdict(t, n)  = J(answer(t, n), gold_fact(t))  in {correct, incorrect}
```

**Formula:**

```
scar_survival_rate  =  |{ t in T : for all n in {1..N}, verdict(t, n) = correct }| / |T|
```

In words: a trap counts as survived **only if the correct answer holds in ALL N
reloads simultaneously**. We divide by the total number of working traps.

### Why "in ALL", not "on average"

Averaging over cycles forgives random failures - which is exactly what we want
to avoid. A cosmetic scar can **accidentally pass one or two cycles** (for
example the instinct happens to phrase the same words as the correct answer)
and that would be folded into the "average" percentage. "In ALL" rejects such
lucky wins: a real scar must win systematically.

Concretely: in experiment 5 (discrimination) the cosmetic regime R2 averaged
10% across cycles (cycle 1 was 100%, cycles 2-10 were 0%), but **scar-survival
= 0%** - no single trap survived all 10 cycles. An average would have lied
here; "in ALL" does not.

---

## 3. Measurement protocol

The steps below must be followed strictly - otherwise the number loses its
meaning.

**Step 1. Selecting the working traps `T`.**
Take a pool of candidate questions. For each candidate, run the model in its
**clean regime** (no notebook, no correction mechanism, typically
`greedy / temperature = 0`). Compare each answer with the reference using the
judge `J`. Keep only the ones the judge labels `incorrect` - those are the
working traps for this model. Without this step the metric is meaningless: if
the model already answers correctly, there is nothing whose "survival" you can
measure.

Cap: no more than ~10 traps per cell - that is enough for stable statistics
and the returns drop sharply beyond that.

**Step 2. Building the notebook `Mem`.**
For every selected trap, record the correct fact in the notebook (external
file). The fact must be **unambiguous** and verifiable; it must not echo the
instinctive wrong answer.

**Step 3. What counts as a "reload".**
A reload `n` is a **separate fresh OS process**:
- the model is fully unloaded from memory (the previous process exits);
- the model is loaded **from scratch** from the on-disk weights (new process);
- the notebook is read again from disk.

In-process "reset" via `del model` or `gc.collect()` **does not count** - it is
weaker and fails to catch hidden state. Only separate processes count.

**Step 4. The "no helping the mechanism" rule.**
Between cycles you **must not**:
- tune any parameters (`alpha`, `beta`, temperature, `max_tokens` length);
- edit the notebook by hand based on the previous cycle's outputs;
- change the prompt or system message;
- memorise "good" phrasings from past cycles.

Every cycle must be an i.i.d. clean reload. Parameter changes are admissible
**only** at the level of experiment design, not between cycles of a single
measurement.

**Step 5. Measurement.**
In every cycle, for every trap `t`, generate `answer(t, n)`; the judge `J`
returns `verdict(t, n)`. Compute the formula from section 2.

---

## 4. How correctness is decided

**Only by an independent semantic judge.** This can be:
- a larger or different model that receives `(question, gold_fact, candidate_answer)`
  and emits a strict `CORRECT/INCORRECT` decision by meaning (in our
  experiments: Qwen2.5-7B);
- a human.

What you **must not** use as the judge:

**Keyword search.** Small models often hallucinate unexpected wrong answers
(their own names, scrambled numbers) that simple keyword matching does not
catch. We saw this in experiment 1: on the 3B model, automatic word-level
scoring under-counted failures by half; the truth only came out via manual
inspection of the answers or via a semantic judge.

**The Confidence Gain metric (probability mass).** Under noise it
**over-estimates resilience**. Specifically: this metric compares the
"correct answer" only against a single canonical "wrong answer", but in a
noisy notebook free-form generation can grab a **third** distractor that was
not part of this comparison. Then Gain reports a high +0.90 / +0.99, but the
actual generated answer is a lie. We logged this in experiment 1 (load 2:
noise): #20 - Gain +0.90, but the model generated "K2"; #40 - Gain +0.37, but
the model generated "Steamboat Willie". The judge marked both `incorrect`.

**Conclusion:** Confidence Gain is kept as an indicator of "how strongly the
instinct is overridden at the token level", but **not** as proof of
correctness. The only honest measure of survival is a judge over the actually
generated answer.

---

## 5. Evidence base for the metric

The metric passed two validation tests.

**Step A - universality (experiment 4, this folder).**
We ran the full protocol on 3 models (Qwen2.5-3B, Qwen2.5-7B,
Phi-3.5-mini-instruct - the last one from a different family, Microsoft) x 2
fact sets (Set A: 58 candidates from myths / geography / quotations; Set B: 20
new geo-facts and world records) = **6 cells x 10 reloads x 7-10 traps = 540
answers**. Result: **scar-survival = 100% across all 6 cells**. The metric is
not pinned to a single configuration and ports across family boundaries.
Technical note: Phi-3.5 in `transformers 5.x` must be loaded **without**
`trust_remote_code` (the upstream `modeling_phi3.py` calls
`DynamicCache.from_legacy_cache`, which was removed in 5.x). Details in
`scar_metric_validation.json` + `scar_metric_validation.png`.

**Step B - discrimination (experiment 5).**
We ran 3 contrastive regimes on Qwen2.5-3B x 14 traps x 10 reloads:

- **R1 (real scar):** mechanism is on every cycle - **100%**.
- **R2 (cosmetic):** notebook only on cycle 1, cycles 2-10 are bare instinct - **0%**.
- **R3 (placebo):** notebook is reattached but contains irrelevant boilerplate text - **14%**.

The 100% vs 0% / 14% gap proves that the metric **measures the work of the
mechanism**, not the model's general competence. Without step B we could not
have been sure that the 100% in step A was not an artefact of "everything
scores 100".

Caveat on R3: 14% is 2 cases where the model refused to answer
("the question cannot be answered based on the provided fact") and the judge
considered the refusals vacuously consistent with the truth. The count of
substantively correct answers under the placebo is zero. Details in
`scar_metric_discrimination.json` + `.png` (experiment 5).

---

## 6. Limits of the metric

**What scar-survival measures.**
- The robustness of a working correction mechanism against **full environment
  reload**.
- Whether the correction is *real* (lives in the external mechanism and
  survives a restart) or *cosmetic* (dissolves with the process state).

**What scar-survival does NOT measure.**

1. **Robustness under pressure.** The metric alone does not say whether the
   mechanism will hold up under notebook noise, ballast from irrelevant
   entries, stochastic generation, or particularly "sticky" instincts. That is
   a separate measurement - our stress test from experiment 1, where exactly
   such a hole was found: under a noisy notebook the survival rate drops from
   100% to 50%. Scar-survival and the stress test measure **different axes**
   of reliability - both are needed.

2. **"Whether the model understands the truth".** No. Our scar lives
   **outside** - in the notebook and in the decoding mechanism, not in the
   model weights. The metric says "the mechanism keeps working after a
   restart", not "the model has internalised the truth". In our architecture
   the model stays frozen and fundamentally does not "understand" - it is
   simply **overridden** at the token-selection level.

3. **Correspondence of the truth to reality.** The metric takes the reference
   fact `gold_fact(t)` as given. If the reference is itself wrong or
   ambiguous, the metric does not notice. In experiment 3 we hit such a case
   (#46 "length of a day on Venus": sidereal != solar) - the trap had to be
   reworded.

---

## 7. The actual scripts that compute the metric

The full chain lives in the repository; below is pseudo-code with pointers to
working files.

```
# 1. Select working traps
for cand in pool:
    plain_answer = model_plain(cand.question)        # baseline_multi.py
    if judge_7B(plain_answer, cand.correct) == "incorrect":
        T.append(cand)                               # judge_baselines.py
T = T[:10]                                           # build_notebooks.py

# 2. Build the notebook
notebook = { t.id: t.authoritative_fact for t in T } # build_notebooks.py

# 3. One cycle = SEPARATE PROCESS
def cycle(n):                                        # scar_cycle_multi.py (exp 4)
    model, tok = load_from_disk()                    # cd_lib_mm.load_any
    answers = []
    for t in T:
        fact = notebook[t.id]                        # from file, not from memory
        p_with    = chat(SYS_NB, f"Authoritative fact: {fact}\n{t.question}")
        p_without = chat(SYS_PLAIN, t.question)
        a = contrastive_generate(model, tok, p_with, p_without, greedy=True)
        answers.append((t.id, a))
    dump(answers, f"cycle_{n}.json")

# 4. Run N reloads (60 = 6 cells x 10) - bash loop, each iter = separate python3
for n in 1..N:
    subprocess.run(["python3", "scar_cycle_multi.py", n])

# 5. Measurement
verdicts = batch_judge_7B(all_cycle_files)           # judge_cycles_multi.py
scar_survival = mean([ all(verdicts[t.id][n] == "correct" for n in 1..N) for t in T ])
                                                     # aggregate_validation.py
```

**Direct pointers to the working implementations:**

- Basic scenario, 1 model x 1 set, 10 cycles: experiment 1 (`exp1-scar/`) -
  `scar_cycle.py` + `judge_all_cycles.py` + `aggregate_scar.py`.
- Universality (step A), 3 models x 2 sets: experiment 4 (this folder) -
  `cd_lib_mm.py`, `baseline_multi.py`, `scar_cycle_multi.py`,
  `judge_baselines.py`, `build_notebooks.py`, `judge_cycles_multi.py`,
  `aggregate_validation.py`.
- Discrimination (step B), 3 regimes x 10 cycles: experiment 5
  (`exp5-discrimination/`) - `discriminate_cycle.py`, `judge_discriminate.py`,
  `aggregate_discriminate.py`, `placebo_notebook.json`.

Reproduction: each script is a self-contained python3 call; data flows between
steps via `.json` files. Launching reloads is a bash loop over
`python3 scar_cycle*.py n`.
