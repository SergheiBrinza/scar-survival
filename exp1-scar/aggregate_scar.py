#!/usr/bin/env python3
"""Aggregates scar-survival: per-trap x cycle (correct yes/no + Confidence Gain),
scar-survival rate per cycle, average Gain per cycle. Renders the plot and writes scar_survival.json.
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
N_CYCLES = 10


def main():
    cycles = {n: json.load(open(os.path.join(HERE, f"cycle_{n}.json"))) for n in range(1, N_CYCLES + 1)}
    judged = json.load(open(os.path.join(HERE, "all_cycles_judged.json")))

    trap_ids = [r["id"] for r in cycles[1]["rows"]]
    questions = {r["id"]: r["question"] for r in cycles[1]["rows"]}

    # per-trap matrices
    correct = {i: [] for i in trap_ids}   # bool per cycle
    gain = {i: [] for i in trap_ids}      # float per cycle
    for n in range(1, N_CYCLES + 1):
        gmap = {r["id"]: r["confidence_gain"] for r in cycles[n]["rows"]}
        vmap = judged[str(n)]["verdicts"]
        for i in trap_ids:
            correct[i].append(vmap[str(i)] == "correct")
            gain[i].append(gmap[i])

    # per cycle
    survival = []   # % correct
    avg_gain = []
    for n in range(1, N_CYCLES + 1):
        c = sum(correct[i][n - 1] for i in trap_ids)
        survival.append(round(100.0 * c / len(trap_ids), 1))
        avg_gain.append(round(sum(gain[i][n - 1] for i in trap_ids) / len(trap_ids), 4))

    # "survived through cycle n" = correct in ALL cycles up to n inclusive
    survived_through = []
    for n in range(1, N_CYCLES + 1):
        s = sum(1 for i in trap_ids if all(correct[i][:n]))
        survived_through.append(round(100.0 * s / len(trap_ids), 1))

    out = {
        "n_cycles": N_CYCLES, "n_traps": len(trap_ids),
        "per_cycle_survival_pct": survival,
        "per_cycle_avg_confidence_gain": avg_gain,
        "survived_through_pct": survived_through,
        "scar_survival_rate_final": survived_through[-1],
        "per_trap": [{
            "id": i, "question": questions[i],
            "correct_each_cycle": correct[i],
            "gain_each_cycle": gain[i],
            "always_correct": all(correct[i]),
        } for i in trap_ids],
    }
    json.dump(out, open(os.path.join(HERE, "scar_survival.json"), "w"), ensure_ascii=False, indent=2)

    # plot
    x = list(range(1, N_CYCLES + 1))
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(x, survived_through, "o-", color="#1a7f37", lw=2.5, label="Scar-survival rate (%)")
    ax1.set_xlabel("Reload number (cycle)")
    ax1.set_ylabel("Traps surviving, %", color="#1a7f37")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.tick_params(axis="y", labelcolor="#1a7f37")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, avg_gain, "s--", color="#8250df", lw=2, label="Average Confidence Gain")
    ax2.set_ylabel("Average Confidence Gain", color="#8250df")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="#8250df")

    plt.title("Scar-survival: mechanism robustness across 10 full reloads\n"
              "(Qwen2.5-3B frozen, notebook + contrastive decoding)")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="lower center")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "scar_survival.png"), dpi=130)
    print("Saved scar_survival.json and scar_survival.png")
    print("survived_through_pct:", survived_through)
    print("per_cycle_avg_confidence_gain:", avg_gain)


if __name__ == "__main__":
    main()
