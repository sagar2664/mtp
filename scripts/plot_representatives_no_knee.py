#!/usr/bin/env python3
"""
Quick helper script to plot representative solution comparison excluding Balanced (Knee).

Reads metrics from `results/experiment_summary.json` and produces
`results/figures/representative_solutions_no_knee.png`.
"""

import json
import pathlib
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results" / "experiment_summary.json"
OUTPUT_PATH = ROOT / "results" / "figures" / "representative_solutions_no_knee.png"


def load_representatives() -> List[Dict[str, float]]:
    with SUMMARY_PATH.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    reps = data.get("representative_solutions", [])
    return [rep for rep in reps if not rep["label"].lower().startswith("balanced (knee")]


def plot_representatives(reps: List[Dict[str, float]]) -> None:
    if not reps:
        raise ValueError("No representative solutions found (after filtering Balanced (Knee)).")

    labels = [rep["label"] for rep in reps]
    cost = [rep["cost"] for rep in reps]
    emissions = [rep["emissions"] for rep in reps]
    resilience = [rep["resilience"] for rep in reps]

    idx = np.arange(len(labels))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_cost = "#1f77b4"
    color_emissions = "#2ca02c"
    color_resilience = "#ff7f0e"

    ax1.bar(idx - width, cost, width, label="Cost ($)", color=color_cost)
    ax1.set_ylabel("Cost ($)", color=color_cost)
    ax1.tick_params(axis="y", labelcolor=color_cost)

    ax2 = ax1.twinx()
    ax2.bar(idx, emissions, width, label="Emissions (kg CO2)", color=color_emissions)
    ax2.set_ylabel("Emissions (kg CO2)", color=color_emissions)
    ax2.tick_params(axis="y", labelcolor=color_emissions)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))
    ax3.bar(idx + width, resilience, width, label="Resilience Score", color=color_resilience)
    ax3.set_ylabel("Resilience Score", color=color_resilience)
    ax3.tick_params(axis="y", labelcolor=color_resilience)

    ax1.set_xticks(idx)
    ax1.set_xticklabels(labels)
    ax1.set_title("Representative Solutions (Excluding Balanced (Knee))")

    lines = []
    for axis in (ax1, ax2, ax3):
        bars = axis.containers[0]
        lines.append(axis.legend([bars], [bars.get_label()], loc="upper left"))

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    reps = load_representatives()
    plot_representatives(reps)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

