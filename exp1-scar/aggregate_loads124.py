#!/usr/bin/env python3
"""Aggregates and plots Loads 1, 2, 4. Writes parts to stress_partial.json and 3 png files."""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load(fn):
    return json.load(open(os.path.join(HERE, fn)))


def main():
    l1 = load("load1_answers_judged.json")
    l2 = load("load2_answers_judged.json")
    l4 = load("load4_answers_judged.json")

    # ---- Load 1: per level ----
    levels = l1["levels"]
    by_lvl = {lv: [r for r in l1["rows"] if r["level"] == lv] for lv in levels}
    l1_surv = [round(100 * sum(r["verdict"] == "correct" for r in by_lvl[lv]) / len(by_lvl[lv]), 1) for lv in levels]
    l1_gain = [round(sum(r["confidence_gain"] for r in by_lvl[lv]) / len(by_lvl[lv]), 4) for lv in levels]
    print("LOAD1 levels:", levels)
    print("  survival%:", l1_surv)
    print("  avg gain :", l1_gain)
    for lv in levels:
        fails = [r["id"] for r in by_lvl[lv] if r["verdict"] != "correct"]
        if fails:
            print(f"  level {lv}: failed {fails}")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(levels, l1_surv, "o-", color="#1a7f37", lw=2.5, label="Survival, %")
    ax1.set_xlabel("Notebook size: unrelated entries (ballast)")
    ax1.set_ylabel("Survived, %", color="#1a7f37"); ax1.set_ylim(0, 105)
    ax1.tick_params(axis="y", labelcolor="#1a7f37"); ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(levels, l1_gain, "s--", color="#8250df", lw=2, label="Average Gain")
    ax2.set_ylabel("Average Confidence Gain", color="#8250df"); ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="#8250df")
    plt.title("Load 1: growing notebook (ballast)\ndoes the signal of the needed fact get diluted")
    l1h, l1l = ax1.get_legend_handles_labels(); l2h, l2l = ax2.get_legend_handles_labels()
    ax1.legend(l1h + l2h, l1l + l2l, loc="lower center")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "load1_ballast.png"), dpi=130); plt.close(fig)

    # ---- Load 2: noise, clean vs noisy ----
    ids = [r["id"] for r in l2["rows"]]
    clean = [r["clean_gain"] for r in l2["rows"]]
    noisy = [r["confidence_gain"] for r in l2["rows"]]
    surv2 = [r["verdict"] == "correct" for r in l2["rows"]]
    n_surv2 = sum(surv2)
    print(f"\nLOAD2 noise survival: {n_surv2}/{len(ids)}")
    for r in l2["rows"]:
        print(f"  #{r['id']:>2} {'OK ' if r['verdict']=='correct' else 'FAIL'} clean={r['clean_gain']:+.3f} noisy={r['confidence_gain']:+.3f}")

    import numpy as np
    x = np.arange(len(ids)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, clean, w, label="clean notebook (Gain)", color="#0969da")
    bars = ax.bar(x + w / 2, noisy, w, label="noisy (Gain)", color="#cf222e")
    for b, ok in zip(bars, surv2):
        b.set_color("#1a7f37" if ok else "#cf222e")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"#{i}" for i in ids])
    ax.set_ylabel("Confidence Gain"); ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Load 2: noisy memory (clean vs noisy)\n"
                 f"green=survived, red=broken; survived {n_surv2}/{len(ids)}")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "load2_noise.png"), dpi=130); plt.close(fig)

    # ---- Load 4: sticky, instinct vs contrastive ----
    ids4 = [r["id"] for r in l4["rows"]]
    pin = [r["P_instinct"] for r in l4["rows"]]
    pcon = [r["P_contrastive"] for r in l4["rows"]]
    surv4 = sum(r["verdict"] == "correct" for r in l4["rows"])
    print(f"\nLOAD4 sticky survival: {surv4}/{len(ids4)}  avg gain={sum(r['confidence_gain'] for r in l4['rows'])/len(ids4):+.3f}")

    x = np.arange(len(ids4)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, pin, w, label="P(correct) instinct", color="#cf222e")
    ax.bar(x + w / 2, pcon, w, label="P(correct) contrastive", color="#1a7f37")
    ax.set_xticks(x); ax.set_xticklabels(ids4, rotation=45, ha="right")
    ax.set_ylabel("P(correct)"); ax.set_ylim(0, 1.05); ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Load 4: sticky traps (instinct ~0 for the correct answer)\n"
                 f"does contrastive override; survived {surv4}/{len(ids4)}")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "load4_sticky.png"), dpi=130); plt.close(fig)

    part = {
        "load1_ballast": {"levels": levels, "survival_pct": l1_surv, "avg_gain": l1_gain,
                          "n_correct": l1["n_correct"], "n_total": l1["n_total"]},
        "load2_noise": {"n_correct": n_surv2, "n_total": len(ids),
                        "per_trap": [{"id": r["id"], "clean_gain": r["clean_gain"],
                                      "noisy_gain": r["confidence_gain"], "survived": r["verdict"] == "correct",
                                      "answer": r["answer"]} for r in l2["rows"]]},
        "load4_sticky": {"n_correct": surv4, "n_total": len(ids4),
                         "avg_gain": round(sum(r["confidence_gain"] for r in l4["rows"]) / len(ids4), 4),
                         "per_trap": [{"id": r["id"], "P_instinct": r["P_instinct"],
                                       "P_contrastive": r["P_contrastive"], "gain": r["confidence_gain"],
                                       "survived": r["verdict"] == "correct"} for r in l4["rows"]]},
    }
    json.dump(part, open(os.path.join(HERE, "stress_partial.json"), "w"), ensure_ascii=False, indent=2)
    print("\nPlots: load1_ballast.png, load2_noise.png, load4_sticky.png; stress_partial.json")


if __name__ == "__main__":
    main()
