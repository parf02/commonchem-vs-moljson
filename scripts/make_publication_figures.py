from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"

SUMMARY_CSV = OUTPUTS_DIR / "summary_by_family.csv"
PAIRED_CSV = OUTPUTS_DIR / "paired_stats.csv"
RESULTS_CSV = OUTPUTS_DIR / "results.csv"

FAMILY_ORDER = [
    "translation_to_graph",
    "translation_graph_to_smiles",
    "shortest_path",
    "constrained_generation",
    "overall",
]
FAMILY_LABELS = {
    "translation_to_graph": "Text -> graph",
    "translation_graph_to_smiles": "Graph -> SMILES",
    "shortest_path": "Shortest path",
    "constrained_generation": "Constrained generation",
    "overall": "Overall",
}
FAMILY_LABELS_COMPACT = {
    "translation_to_graph": "Text\n-> graph",
    "translation_graph_to_smiles": "Graph\n-> SMILES",
    "shortest_path": "Shortest\npath",
    "constrained_generation": "Constrained\ngeneration",
    "overall": "Overall",
}

MOLJSON_COLOR = "#305c8c"
COMMONCHEM_COLOR = "#c65f2d"
GRID_COLOR = "#d9dde3"
TEXT_COLOR = "#1f2430"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "grid.linestyle": "-",
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUTS_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUTS_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def make_accuracy_figure(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df[summary_df["family"].isin(FAMILY_ORDER)].copy()
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.84)

    x = np.arange(len(FAMILY_ORDER))
    width = 0.34

    for offset, representation, color, label in [
        (-width / 2, "moljson", MOLJSON_COLOR, "MolJSON"),
        (width / 2, "commonchem", COMMONCHEM_COLOR, "CommonChem"),
    ]:
        sub = plot_df[plot_df["representation"] == representation].set_index("family").loc[FAMILY_ORDER]
        y = 100.0 * sub["accuracy"].to_numpy(dtype=float)
        lower = y - 100.0 * sub["ci_low"].to_numpy(dtype=float)
        upper = 100.0 * sub["ci_high"].to_numpy(dtype=float) - y
        ax.bar(
            x + offset,
            y,
            width=width,
            color=color,
            label=label,
            yerr=np.vstack([lower, upper]),
            capsize=4,
            linewidth=0,
            zorder=3,
        )

    ax.axvspan(3.5, 4.5, color="#f4f6f8", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS_COMPACT[f] for f in FAMILY_ORDER], rotation=0, ha="center")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    fig.suptitle("Accuracy by task family", y=0.975, fontsize=12)
    fig.legend(
        handles=[
            Patch(facecolor=MOLJSON_COLOR, label="MolJSON"),
            Patch(facecolor=COMMONCHEM_COLOR, label="CommonChem"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
    )
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.text(4.0, 103.0, "Overall", fontsize=9, ha="center", va="bottom", color="#666666")

    save_figure(fig, "figure_1_accuracy_by_task")


def make_outcomes_figure(results_df: pd.DataFrame) -> None:
    rows: list[dict[str, float | str]] = []
    for family in FAMILY_ORDER:
        family_df = results_df if family == "overall" else results_df[results_df["family"] == family]
        pivot = (
            family_df.pivot_table(index="pair_id", columns="representation", values="is_correct", aggfunc="first")
            .dropna()
            .astype(int)
        )
        both_correct = int(((pivot["commonchem"] == 1) & (pivot["moljson"] == 1)).sum())
        cc_only = int(((pivot["commonchem"] == 1) & (pivot["moljson"] == 0)).sum())
        mj_only = int(((pivot["commonchem"] == 0) & (pivot["moljson"] == 1)).sum())
        both_wrong = int(((pivot["commonchem"] == 0) & (pivot["moljson"] == 0)).sum())
        total = int(len(pivot))
        rows.append(
            {
                "family": family,
                "both_correct": 100.0 * both_correct / total,
                "commonchem_only": 100.0 * cc_only / total,
                "moljson_only": 100.0 * mj_only / total,
                "both_wrong": 100.0 * both_wrong / total,
            }
        )

    plot_df = pd.DataFrame(rows).set_index("family").loc[FAMILY_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.14, top=0.84)
    y = np.arange(len(plot_df))

    left = np.zeros(len(plot_df))
    segments = [
        ("both_correct", "#86b875", "Both correct"),
        ("commonchem_only", COMMONCHEM_COLOR, "CommonChem only"),
        ("moljson_only", MOLJSON_COLOR, "MolJSON only"),
        ("both_wrong", "#d9dde3", "Both wrong"),
    ]
    for key, color, label in segments:
        values = plot_df[key].to_numpy(dtype=float)
        ax.barh(y, values, left=left, color=color, label=label, edgecolor="white", linewidth=0.8)
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels([FAMILY_LABELS[f] for f in plot_df["family"]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of paired prompts (%)")
    fig.suptitle("How the paired outcomes break down", y=0.975, fontsize=12)
    fig.legend(
        handles=[
            Patch(facecolor="#86b875", label="Both correct"),
            Patch(facecolor=COMMONCHEM_COLOR, label="CommonChem only"),
            Patch(facecolor=MOLJSON_COLOR, label="MolJSON only"),
            Patch(facecolor="#d9dde3", label="Both wrong"),
        ],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.95),
    )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()

    save_figure(fig, "figure_3_paired_outcomes")


def main() -> int:
    setup_style()
    summary_df = pd.read_csv(SUMMARY_CSV)
    results_df = pd.read_csv(RESULTS_CSV)

    make_accuracy_figure(summary_df)
    make_outcomes_figure(results_df)
    print(OUTPUTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
