#!/usr/bin/env python3
"""Aggregates Load 3 (stochastic): spread of the correct-answer win share across 14 traps
over 10 cycles. Renders load3_stochastic.png (mean +/- per-cycle range)."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    d = json.load(open(os.path.join(HERE, "stoch_judged.json")))
    trap_ids = d["trap_ids"]
    spread = d["per_trap_spread"]  # tid(str) -> [10 fracs]

    means, mins, maxs = [], [], []
    for tid in trap_ids:
        arr = np.array(spread[str(tid)])
        means.append(arr.mean()); mins.append(arr.min()); maxs.append(arr.max())
    means = np.array(means); mins = np.array(mins); maxs = np.array(maxs)

    x = np.arange(len(trap_ids))
    fig, ax = plt.subplots(figsize=(11, 5))
    yerr = np.vstack([means - mins, maxs - means])
    ax.errorbar(x, means, yerr=yerr, fmt="o", color="#1a7f37", ecolor="#8250df",
                elinewidth=2, capsize=5, ms=7, label="correct-answer win share (mean, per-cycle range across 10 cycles)")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"#{i}" for i in trap_ids])
    ax.set_ylabel("Fraction of runs where the correct answer won"); ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    overall = float(np.mean(d["per_cycle_winrate"]))
    ax.set_title("Load 3: stochastic generation (temperature=0.7), contrastive\n"
                 f"spread across 14 traps over 10 reloads; overall win share {overall:.2f}")
    ax.legend(loc="lower center")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "load3_stochastic.png"), dpi=130)
    print("per_cycle_winrate:", d["per_cycle_winrate"])
    print("overall winrate:", round(overall, 3))
    # traps with notable spread
    for tid in trap_ids:
        arr = spread[str(tid)]
        if min(arr) < 0.8:
            print(f"  #{tid}: spread {arr}")
    print("Saved load3_stochastic.png")


if __name__ == "__main__":
    main()
