"""Export a completed predictor run's ensemble-size comparison as PNG/CSV/MD.

uv run python scripts/predictor_ensemble_table.py <downloaded-run-dir>
"""

import argparse
import csv
import json
from pathlib import Path


def build_table(run: Path) -> Path:
    summary = json.loads(
        (run / "predictor_ensemble_summary.json").read_text(encoding="utf-8")
    )
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    training = json.loads((run / "predictor/training.json").read_text(encoding="utf-8"))
    n_train = len(training["train_indices"])
    source = {"creilov_combined": "S1 + S2", "creilov": "S2"}.get(
        Path(cfg["csv_path"]).stem, Path(cfg["csv_path"]).stem
    )
    model = {
        "one_hot_mlp": "one-hot MLP",
        "linear_probe": "linear probe",
        "additive": "additive",
    }[cfg["predictor"]["kind"]]
    sizes = cfg["predictor_ensemble_sizes"]
    if set(summary) != set(map(str, sizes)):
        raise ValueError("Every requested ensemble must have completed evaluation")
    index = {
        (int(size), row["mask_fraction"]): row
        for size, entry in summary.items()
        for row in entry["reports"]
    }
    if any(row["partition"] != "validation" for row in index.values()):
        raise ValueError("Expected validation diagnostics, not test results")
    headers = [
        "Members",
        "Clean ρ",
        "50% mask ρ",
        "90% mask ρ",
        "Clean RMSE",
        "50% mask RMSE",
        "90% mask RMSE",
    ]
    rows = [
        [str(n)]
        + [
            (
                "—"
                if index[(n, mask)][metric] is None
                else f"{index[(n, mask)][metric]:.3f}"
            )
            for metric in ("spearman", "rmse")
            for mask in (0.0, 0.5, 0.9)
        ]
        for n in sizes
    ]
    output = run / "ensemble_comparison.png"
    with (run / "ensemble_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "ensemble_size",
            "mask_fraction",
            "n",
            "rmse",
            "mae",
            "spearman",
            "brier",
            "gaussian_nll",
            "coverage_95",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for n in sizes:
            for mask in (0.0, 0.5, 0.9):
                row = index[(n, mask)]
                writer.writerow({"ensemble_size": n, **{k: row[k] for k in fields[1:]}})
    markdown = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    markdown += "\n".join("| " + " | ".join(row) + " |" for row in rows)
    note = "Validation data selects checkpoints and calibrates variance; these are not independent test metrics. Sizes share nested members and training rows."
    (run / "ensemble_comparison.md").write_text(
        markdown + "\n\n" + note + "\n", encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    fig.subplots_adjust(left=0.035, right=0.965, top=0.73, bottom=0.23)
    fig.text(
        0.035,
        0.94,
        "CreiLOV predictor ensemble comparison",
        fontsize=22,
        fontweight="bold",
        color="#172b45",
        va="top",
    )
    count = index[(sizes[0], 0.0)]["n"]
    fig.text(
        0.035,
        0.82,
        f"{source} · {model} · {n_train:,} training sequences · {count:,} validation sequences · seed {cfg['seed']}",
        fontsize=11,
        color="#526278",
    )
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="right",
        colLoc="right",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.auto_set_column_width(list(range(len(headers))))
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dce3eb")
        cell.set_linewidth(0.6)
        cell.PAD = 0.1
        if row == 0:
            cell.set_facecolor("#172b45")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f0f5fa" if row % 2 else "white")
        if col == 0:
            cell.set_text_props(ha="left")
    fig.text(
        0.035,
        0.13,
        "Spearman ρ: higher is better. RMSE: lower is better, in log10 fluorescence units.\nValidation diagnostics used for checkpoint selection/calibration; single seed, nested members, no confidence intervals.",
        fontsize=10,
        color="#526278",
        va="top",
        linespacing=1.5,
    )
    fig.savefig(output, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), dpi=180)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.065, right=0.975, top=0.72, bottom=0.23, wspace=0.24)
    fig.text(
        0.065,
        0.95,
        "Do more predictor members improve performance?",
        fontsize=22,
        fontweight="bold",
        color="#172b45",
        va="top",
    )
    fig.text(
        0.065,
        0.86,
        f"CreiLOV {source} · {n_train:,} training sequences · {count:,} validation sequences · seed {cfg['seed']}",
        fontsize=11,
        color="#526278",
    )
    colors = ["#0072B2", "#D55E00", "#009E73"]
    width = 0.72 / len(sizes)
    for ax, metric, title, ylabel in (
        (axes[0], "spearman", "Rank correlation ↑", "Spearman ρ"),
        (axes[1], "rmse", "Prediction error ↓", "RMSE (log10 fluorescence)"),
    ):
        for j, size in enumerate(sizes):
            x = [i + (j - (len(sizes) - 1) / 2) * width for i in range(3)]
            values = [index[(size, fraction)][metric] for fraction in (0.0, 0.5, 0.9)]
            bars = ax.bar(
                x,
                values,
                width=width * 0.92,
                color=colors[j % len(colors)],
                label=f"{size} member{'s' if size != 1 else ''}",
            )
            ax.bar_label(
                bars, labels=[f"{v:.3f}" for v in values], padding=4, fontsize=9
            )
        ax.set_title(
            title, loc="left", fontsize=15, fontweight="bold", pad=18, color="#172b45"
        )
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks([0, 1, 2], ["0% (clean)", "50%", "90%"])
        ax.set_xlabel("Masked residue positions", fontsize=11, labelpad=10)
        ax.set_ylim(
            0,
            (
                1.05
                if metric == "spearman"
                else max(r["rmse"] for r in index.values()) * 1.2
            ),
        )
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.975, 0.85),
        ncol=len(sizes),
        frameon=False,
    )
    fig.text(
        0.065,
        0.11,
        "Validation diagnostics: these data selected checkpoints and calibrated variance. The test set remains untouched.\nEnsembles share training rows and nested members. One seed and one masking draw per level; no confidence intervals.",
        fontsize=10,
        color="#526278",
        va="top",
        linespacing=1.5,
    )
    fig.savefig(run / "ensemble_performance.png", facecolor="white")
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    print(build_table(parser.parse_args().run_dir))
