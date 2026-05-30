#!/usr/bin/env python3
"""Aggregate the scar-survival metric across 6 cells (model x set).

Writes scar_metric_validation.json + png.
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3.5-mini-instruct"]
SETS = ["setA", "setB"]


def short(m):
    return m.split("/")[-1].replace(".", "_")


def main():
    judged = json.load(open(os.path.join(HERE, "cycles_judged.json")))
    nb = {(short(m), s): json.load(open(os.path.join(HERE, f"notebook_{short(m)}_{s}.json")))
          for m in MODELS for s in SETS
          if os.path.exists(os.path.join(HERE, f"notebook_{short(m)}_{s}.json"))}
    summary_rows = []
    cells_detail = {}
    for m in MODELS:
        for s in SETS:
            cell = f"{short(m)}/{s}"
            if cell not in judged:
                continue
            traps = nb[(short(m), s)]["traps"]
            tids = [t["id"] for t in traps]
            per_cycle = judged[cell]
            cycles = sorted(map(int, per_cycle.keys()))
            # survival across cycles
            per_cycle_pct = []
            survived_through = []  # fraction of traps correct across ALL cycles 1..n
            always_correct = {tid: True for tid in tids}
            for n in cycles:
                cyc = per_cycle[str(n)]
                ok = sum(1 for tid in tids if cyc.get(str(tid)) == "correct")
                per_cycle_pct.append(round(100 * ok / len(tids), 1) if tids else 0)
                for tid in tids:
                    if cyc.get(str(tid)) != "correct":
                        always_correct[tid] = False
                s_through = sum(1 for tid in tids if always_correct[tid])
                survived_through.append(round(100 * s_through / len(tids), 1) if tids else 0)
            cells_detail[cell] = {"n_traps": len(tids), "trap_ids": tids,
                                  "cycles": cycles,
                                  "per_cycle_correct_pct": per_cycle_pct,
                                  "survived_through_pct": survived_through,
                                  "scar_survival_final": survived_through[-1] if survived_through else None}
            summary_rows.append({"model": m, "set": s, "cell": cell,
                                 "n_traps": len(tids),
                                 "scar_survival_final_pct": survived_through[-1] if survived_through else None,
                                 "mean_per_cycle_pct": round(sum(per_cycle_pct) / max(1, len(per_cycle_pct)), 1)})

    out = {"models": MODELS, "sets": SETS, "summary": summary_rows, "cells_detail": cells_detail}
    json.dump(out, open(os.path.join(HERE, "scar_metric_validation.json"), "w"), ensure_ascii=False, indent=2)

    # console table
    print("Scar-survival TABLE per cell:")
    print(f"{'model':>32}  {'set':>5}  {'N':>3}  {'scar%':>5}  {'mean/cycle':>11}")
    for r in summary_rows:
        print(f"{short(r['model']):>32}  {r['set']:>5}  {r['n_traps']:>3}  "
              f"{r['scar_survival_final_pct']:>5}  {r['mean_per_cycle_pct']:>11}")

    # plot: bars across 6 cells, scar_survival_final
    labels = [f"{short(r['model'])}\n{r['set']}\n(N={r['n_traps']})" for r in summary_rows]
    vals = [r["scar_survival_final_pct"] for r in summary_rows]
    cols = ["#1a7f37" if v == 100 else "#bf8700" if v >= 70 else "#cf222e" for v in vals]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(labels)), vals, color=cols, edgecolor="black")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}%", ha="center", fontweight="bold")
    ax.axhline(100, color="gray", ls=":", lw=1)
    ax.set_ylim(0, 112); ax.set_ylabel("Scar-survival rate, %")
    ax.set_title("Scar-survival metric validation: 3 models x 2 fact sets\n"
                 "(10 reloads, contrastive decoding, 7B judge over the answer)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "scar_metric_validation.png"), dpi=130)
    print("\nSaved scar_metric_validation.json and scar_metric_validation.png")


if __name__ == "__main__":
    main()
