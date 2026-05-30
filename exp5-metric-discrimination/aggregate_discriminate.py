#!/usr/bin/env python3
"""Aggregate scar-survival across 3 regimes. Writes scar_metric_discrimination.json + png."""
import os
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OP1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exp1-scar"))
sys.path.insert(0, OP1)

HERE = os.path.dirname(os.path.abspath(__file__))
REGIMES = ["R1", "R2", "R3"]
LABELS = {"R1": "R1: real scar\n(notebook + contrastive,\nevery cycle)",
          "R2": "R2: cosmetic\n(only cycle 1 with notebook,\nthen bare instinct)",
          "R3": "R3: placebo\n(notebook with irrelevant\nentry, contrastive idling)"}


def main():
    judged = json.load(open(os.path.join(HERE, "disc_judged.json")))
    traps = json.load(open(os.path.join(OP1, "working_set_3b.json")))["traps"]
    tids = [t["id"] for t in traps]
    n_traps = len(tids)

    results = {}
    for R in REGIMES:
        cyc = judged.get(R, {})
        cycles = sorted(map(int, cyc.keys()))
        per_cycle_pct = []
        always = {tid: True for tid in tids}
        survived_through = []
        for n in cycles:
            cycdict = cyc[str(n)]
            ok = sum(1 for tid in tids if cycdict.get(str(tid)) == "correct")
            per_cycle_pct.append(round(100 * ok / n_traps, 1))
            for tid in tids:
                if cycdict.get(str(tid)) != "correct":
                    always[tid] = False
            survived_through.append(round(100 * sum(always.values()) / n_traps, 1))
        results[R] = {"per_cycle_correct_pct": per_cycle_pct,
                      "survived_through_pct": survived_through,
                      "scar_survival_final": survived_through[-1] if survived_through else None,
                      "mean_per_cycle_pct": round(sum(per_cycle_pct)/max(1,len(per_cycle_pct)), 1)}

    out = {"model": "Qwen/Qwen2.5-3B-Instruct", "n_traps": n_traps, "n_cycles": 10,
           "regimes": results,
           "discrimination_verdict": (
               "metric_discriminates" if results["R1"]["scar_survival_final"] >= 90
                                       and results["R2"]["scar_survival_final"] <= 30
                                       and results["R3"]["scar_survival_final"] <= 30
               else "ambiguous")}
    json.dump(out, open(os.path.join(HERE, "scar_metric_discrimination.json"), "w"),
              ensure_ascii=False, indent=2)

    # table
    print(f"{'regime':<6}  {'scar%':>5}  {'mean/cycle':>13}  {'per-cycle (10)':<40}")
    for R in REGIMES:
        r = results[R]
        print(f"{R:<6}  {r['scar_survival_final']:>5}  {r['mean_per_cycle_pct']:>13}  {r['per_cycle_correct_pct']}")

    # plot
    labels = [LABELS[R] for R in REGIMES]
    vals = [results[R]["scar_survival_final"] for R in REGIMES]
    cols = ["#1a7f37", "#cf222e", "#cf222e"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, vals, color=cols, edgecolor="black")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}%",
                ha="center", fontweight="bold")
    ax.axhline(100, color="gray", ls=":", lw=1)
    ax.set_ylim(0, 112); ax.set_ylabel("Scar-survival rate, %")
    ax.set_title("Scar-survival metric discrimination: real scar vs fakes\n"
                 "(Qwen2.5-3B, 14 traps, 10 reloads, 7B judge on answer)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "scar_metric_discrimination.png"), dpi=130)
    print("\nSaved scar_metric_discrimination.json and scar_metric_discrimination.png")


if __name__ == "__main__":
    main()
