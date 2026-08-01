#!/usr/bin/env python3
"""Regenerate paper figures 1-5 in English from the canonical n=14 raw run.

Every value is read from a raw experiment artefact; nothing is hard-coded.
fig6_thin_channel.png is already English and canon-correct and is not touched
here (it is produced by опыт6_тонкий_канал/run_frequency_sweep.py).

Usage:  python3 make_figs.py [--check]
  --check  print the values that would be plotted, write nothing
"""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP1 = "/home/softer/вторая_петля/опыт1_шрам"
EXP2 = "/home/softer/публикация/external-grounding/exp2-guardian-v1-model-arbiter"
EXP3 = "/home/softer/публикация/external-grounding/exp3-guardian-v2-wikipedia"
EXP5 = "/home/softer/публикация/scar-survival/exp5-metric-discrimination"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")

GREEN, RED, GOLD, PURPLE = "#1a7f37", "#cf222e", "#bf8700", "#8250df"
CHECK = "--check" in sys.argv


def load(p):
    with open(p) as fp:
        return json.load(fp)


def save(fig, name):
    if CHECK:
        plt.close(fig)
        return
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def bars(ax, labels, vals, colors, fmt="{:.0f}%"):
    bb = ax.bar(labels, vals, color=colors, edgecolor="black")
    for b in bb:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                fmt.format(b.get_height()), ha="center", fontweight="bold")
    ax.axhline(100, color="gray", ls=":", lw=1)
    ax.set_ylim(0, 112)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)


# --------------------------------------------------------------- fig 1
surv = load(f"{EXP1}/scar_survival.json")
n_traps, n_cycles = surv["n_traps"], surv["n_cycles"]
pct = surv["per_cycle_survival_pct"]
gain = surv["per_cycle_avg_confidence_gain"]
print(f"fig1: n_traps={n_traps} cycles={n_cycles} survival={set(pct)} gain={set(gain)}")

fig, ax = plt.subplots(figsize=(9, 5))
x = list(range(1, n_cycles + 1))
ax.plot(x, pct, "o-", color=GREEN, lw=2, ms=7, label="Scar-survival rate (%)")
ax.set_ylim(0, 105)
ax.set_xticks(x)
ax.set_xlabel("Reload number (cycle)")
ax.set_ylabel("Traps surviving, %", color=GREEN)
ax.tick_params(axis="y", labelcolor=GREEN)
ax.grid(True, alpha=0.3)
ax.set_axisbelow(True)
ax2 = ax.twinx()
ax2.plot(x, gain, "s--", color=PURPLE, lw=2, ms=6, label="Mean Confidence Gain")
ax2.set_ylim(0, 1.05)
ax2.set_ylabel("Mean Confidence Gain", color=PURPLE)
ax2.tick_params(axis="y", labelcolor=PURPLE)
ax.set_title(f"Scar survival across {n_cycles} full process reloads\n"
             f"(Qwen2.5-3B frozen, notebook + contrastive decoding, "
             f"{n_traps} working traps)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower center", framealpha=0.95)
save(fig, "fig1_scar_survival.png")

# --------------------------------------------------------------- fig 2
st = load(f"{EXP1}/stress_test.json")
v_base = st["baseline_clean_reload"]["survival_pct"]
v_ball = st["load1_ballast"]["worst_survival_pct"]
v_noise = st["load2_noise"]["survival_pct"]
v_stoch = st["load3_stochastic"]["win_rate"] * 100
v_stick = st["load4_sticky"]["survival_pct"]
print(f"fig2: base={v_base} ballast={v_ball} noise={v_noise} "
      f"stoch={v_stoch} sticky={v_stick}")

fig, ax = plt.subplots(figsize=(10, 5.5))
labels = ["Baseline\n(clean,\n10 reloads)", "Load 1\nballast\n(+500)",
          "Load 2\nnoisy\nnotebook", "Load 3\nstochastic\n(temp 0.7)",
          "Load 4\nsticky\ninstincts"]
vals = [v_base, v_ball, v_noise, v_stoch, v_stick]
cols = [GREEN, GREEN, RED, GREEN, GREEN]
bb = ax.bar(labels, vals, color=cols, edgecolor="black")
for b, v in zip(bb, vals):
    txt = f"{v:.1f}%" if abs(v - round(v)) > 1e-9 else f"{v:.0f}%"
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, txt,
            ha="center", fontweight="bold")
ax.axhline(100, color="gray", ls=":", lw=1)
ax.axhline(50, color="gray", ls=":", lw=1)
ax.set_ylim(0, 112)
ax.grid(True, axis="y", alpha=0.3)
ax.set_axisbelow(True)
ax.set_ylabel("Survival / win rate, %")
ax.set_title("Wear stress test: where the mechanism holds and where it breaks\n"
             f"(Qwen2.5-3B frozen, notebook + contrastive decoding, "
             f"{n_traps} working traps)")
save(fig, "fig2_stress.png")

# --------------------------------------------------------------- fig 3
md = load(f"{EXP2}/memory_defense.json")
n2 = md["n_total"]
v_sick = md["sick_no_defense"]["pct"]
v_prot = md["protected_gatekeeper"]["pct"]
kept_n, kept_d = md["correct_fact_kept_by_arbiter"].split("/")
v_ceil = round(100.0 * int(kept_n) / int(kept_d), 1)
print(f"fig3: n={n2} sick={v_sick} protected={v_prot} "
      f"ceiling={v_ceil} (arbiter kept {md['correct_fact_kept_by_arbiter']})")
assert v_prot == v_ceil, (v_prot, v_ceil)

fig, ax = plt.subplots(figsize=(8, 5))
bars(ax, [f"Sick\n(no defense)\n{md['sick_no_defense']['correct']}/{n2}",
          f"Protected\n(gatekeeper)\n{md['protected_gatekeeper']['correct']}/{n2}",
          f"Defense ceiling\n(arbiter's own accuracy)\n{kept_n}/{kept_d}"],
     [v_sick, v_prot, v_ceil], [RED, GOLD, GREEN], fmt="{:.1f}%")
ax.set_ylabel("Survival (7B judge over the answer), %")
ax.set_title("Guardian 1.0 — a same-family arbiter (Experiment 2)\n"
             "the defense ceiling equals the accuracy of the verifier")
save(fig, "fig3_guardian_v1.png")

# --------------------------------------------------------------- fig 4
g23 = load(f"{EXP3}/gatekeeper23_results.json")
arc = g23["pct_arc"]
n3 = g23["n"]
counts = load(f"{EXP3}/gatekeeper22_results.json")["counts"]
counts = dict(counts, gk23=g23["gk23_correct"])
order = ["sick", "gk1", "gk2", "gk21", "gk22", "gk23"]
print(f"fig4: n={n3} arc=" + " -> ".join(f"{arc[k]}" for k in order))
assert [arc[k] for k in order] == [50.0, 64.3, 71.4, 71.4, 85.7, 100.0], arc

fig, ax = plt.subplots(figsize=(12, 5.5))
labels = [f"Sick\nno defense\n{counts['sick']}/{n3}",
          f"1.0\nclone\narbiter\n{counts['gk1']}/{n3}",
          f"2.0\nWikipedia\n{counts['gk2']}/{n3}",
          f"2.1\nbrute\nstrengthening\n{counts['gk21']}/{n3}",
          f"2.2\ntargeted\nfixes\n{counts['gk22']}/{n3}",
          f"2.3\nfinal\ncalibration\n{counts['gk23']}/{n3}"]
vals = [arc[k] for k in order]
cols = [RED, GOLD, "#2da44e", "#2da44e", "#57c46a", GREEN]
bars(ax, labels, vals, cols, fmt="{:.1f}%")
ax.axhline(arc["gk1"], color=GOLD, ls="--", lw=1)
ax.text(0.01, arc["gk1"] - 3.5, "clone-arbiter ceiling", color=GOLD,
        fontsize=9, transform=ax.get_yaxis_transform(), ha="left", va="top")
# mark the one flat step
ax.annotate("", xy=(3, 80), xytext=(2, 80),
            arrowprops=dict(arrowstyle="<->", color="gray", lw=1))
ax.text(2.5, 81.5, "flat step:\nbrute strengthening adds nothing",
        ha="center", fontsize=8.5, color="dimgray", linespacing=1.3)
ax.set_ylabel("Survival (7B judge over the answer), %")
ax.set_title("Full survival arc under a polluted notebook (Experiments 1-3)\n"
             f"50% -> 100% over {n3} working traps")
save(fig, "fig4_guardian_v2_arc.png")

# --------------------------------------------------------------- fig 5
disc = load(f"{EXP5}/scar_metric_discrimination.json")
n5, c5 = disc["n_traps"], disc["n_cycles"]
r = disc["regimes"]
vals = [r["R1"]["scar_survival_final"], r["R2"]["scar_survival_final"],
        r["R3"]["scar_survival_final"]]
print(f"fig5: n={n5} cycles={c5} R1={vals[0]} R2={vals[1]} R3={vals[2]}")
assert vals == [100.0, 0.0, 14.3], vals

fig, ax = plt.subplots(figsize=(10, 5.5))
labels = ["R1: real scar\n(notebook + contrastive,\nevery cycle)",
          "R2: cosmetic\n(notebook on cycle 1 only,\nthen bare instinct)",
          "R3: placebo\n(irrelevant notebook entry,\ncontrastive on nothing)"]
bars(ax, labels, vals, [GREEN, RED, RED], fmt="{:.1f}%")
ax.set_ylabel("Scar-survival rate, %")
ax.set_title("Discrimination of the scar-survival metric: real scar vs. fakes\n"
             f"(Qwen2.5-3B, {n5} traps, {c5} reloads, 7B judge over the answer)")
save(fig, "fig5_discrimination.png")

print("\nfig6_thin_channel.png: left untouched (already English, canon-correct;"
      " produced by опыт6_тонкий_канал/run_frequency_sweep.py)")
if CHECK:
    print("\n--check: nothing written")
