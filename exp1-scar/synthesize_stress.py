#!/usr/bin/env python3
"""Final stress-test summary: combines 4 loads + the baseline (deterministic reload)
into stress_test.json and renders the master plot stress_master.png."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    part = json.load(open(os.path.join(HERE, "stress_partial.json")))
    stoch = json.load(open(os.path.join(HERE, "stoch_judged.json")))

    l1 = part["load1_ballast"]
    l2 = part["load2_noise"]
    l4 = part["load4_sticky"]
    l3_winrate = float(np.mean(stoch["per_cycle_winrate"]))

    summary = {
        "baseline_clean_reload": {"survival_pct": 100.0, "note": "sub-step 5b + scar-survival: 14/14, 10 reloads"},
        "load1_ballast": {
            "survival_pct_by_level": dict(zip(map(str, l1["levels"]), l1["survival_pct"])),
            "avg_gain_by_level": dict(zip(map(str, l1["levels"]), l1["avg_gain"])),
            "worst_survival_pct": min(l1["survival_pct"]),
            "verdict": "barely affected: only #16 falls; Gain stable ~0.65-0.72",
        },
        "load2_noise": {
            "survival_pct": round(100 * l2["n_correct"] / l2["n_total"], 1),
            "n": f"{l2['n_correct']}/{l2['n_total']}",
            "failed_traps": [t["id"] for t in l2["per_trap"] if not t["survived"]],
            "verdict": "FAILURE: half of the traps break; model copies a distractor; Gain occasionally goes negative",
        },
        "load3_stochastic": {
            "win_rate": round(l3_winrate, 3),
            "per_cycle_winrate": stoch["per_cycle_winrate"],
            "verdict": "holds: at temperature=0.7 the correct answer wins in 100% of samples, no spread",
        },
        "load4_sticky": {
            "survival_pct": round(100 * l4["n_correct"] / l4["n_total"], 1),
            "n": f"{l4['n_correct']}/{l4['n_total']}",
            "avg_gain": l4["avg_gain"],
            "verdict": "holds: contrastive overrides even a near-immovable instinct (Gain +0.58)",
        },
        "conclusion": "The only failure zone is Load 2 (conflicting / noisy memory). "
                      "Ballast, stochasticity, and instinct stickiness are all handled. "
                      "The robustness ceiling is the quality of the notebook, not its size, "
                      "randomness, or the strength of the instinct.",
    }
    json.dump(summary, open(os.path.join(HERE, "stress_test.json"), "w"), ensure_ascii=False, indent=2)

    # master plot
    labels = ["Baseline\n(clean,\n10 reloads)", "Load 1\nballast\n(+500)", "Load 2\nnoise",
              "Load 3\nstochastic\n(temp 0.7)", "Load 4\nsticky\ninstincts"]
    vals = [100.0, min(l1["survival_pct"]), summary["load2_noise"]["survival_pct"],
            round(l3_winrate * 100, 1), summary["load4_sticky"]["survival_pct"]]
    colors = ["#1a7f37", "#1a7f37", "#cf222e", "#1a7f37", "#1a7f37"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold")
    ax.axhline(50, color="gray", ls=":", lw=1)
    ax.set_ylabel("Survival / correct-answer win share, %")
    ax.set_ylim(0, 110)
    ax.set_title("Stress test: where the mechanism breaks\n"
                 "(Qwen2.5-3B frozen, notebook + contrastive decoding)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "stress_master.png"), dpi=130)

    print("=== STRESS TEST SUMMARY ===")
    print(f"Baseline (clean):     100%  (14/14, 10 reloads)")
    print(f"Load 1 ballast +500:  {min(l1['survival_pct'])}%  (Gain per level {l1['avg_gain']})")
    print(f"Load 2 noise:         {summary['load2_noise']['survival_pct']}%  ({l2['n_correct']}/{l2['n_total']}, failed {summary['load2_noise']['failed_traps']})")
    print(f"Load 3 stochastic:    {round(l3_winrate*100,1)}%  (win share, no spread)")
    print(f"Load 4 sticky:        {summary['load4_sticky']['survival_pct']}%  (12/12, Gain +{l4['avg_gain']})")
    print("Saved stress_test.json, stress_master.png")


if __name__ == "__main__":
    main()
