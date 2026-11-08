"""Summarize a benchmark run's Spearman correlations as Markdown, CSV, and PNG.

Reads ``benchmark/summary.json`` from a completed ``creilov benchmark`` run
(local or downloaded from the Modal ``creilov-runs`` Volume) and writes a
compact per-split table next to it.

    uv run --with matplotlib python scripts/spearman_table.py outputs/workflow

Pass ``--no-png`` to skip the image and avoid the matplotlib dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SPLITS = ("random", "distance", "activity", "position")
# (column label, role, mask fraction) -- mask 0.0 is the clean, unmasked input.
COLUMNS = (
    ("Oracle (clean)", "oracle", 0.0),
    ("Predictor (clean)", "predictor", 0.0),
    ("Predictor (50% masked)", "predictor", 0.5),
    ("Predictor (90% masked)", "predictor", 0.9),
)

INK = "#1b1b1f"
MUTED = "#5b5f66"
RULE = "#c9ced6"
ACCENT = "#eef1f6"


def build_rows(run: Path) -> tuple[list[str], list[list[str]], dict]:
    entries = json.loads((run / "benchmark" / "summary.json").read_text())
    sizes = json.loads((run / "split_summary.json").read_text())
    config = json.loads((run / "config.json").read_text())

    for entry in entries:
        if entry["partition"] != "test":
            raise ValueError(f"expected test partition, got {entry['partition']!r}")
        if entry["used_for_checkpoint_selection_and_calibration"]:
            raise ValueError("summary contains a selection/calibration partition")

    index = {(e["split"], e["role"], e["mask_fraction"]): e for e in entries}
    header = ["Split", "Test variants", *(c[0] for c in COLUMNS)]
    rows = []
    for split in SPLITS:
        row = [split.capitalize(), f"{sizes[split]['test']:,}"]
        for _, role, mask in COLUMNS:
            entry = index.get((split, role, mask))
            value = None if entry is None else entry["spearman"]
            # Spearman is JSON null when a partition has no rank variance.
            row.append("n/a" if value is None else f"{value:.3f}")
        rows.append(row)
    return header, rows, config


def write_markdown(out: Path, header, rows, config, run_name: str) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]

    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    oracle = config["oracle"]
    text = [
        "# CreiLOV benchmark: Spearman correlation",
        "",
        f"Run `{run_name}` ({config['device'].upper()}). Spearman rank correlation "
        "between predicted and measured log10 fluorescence on each split's "
        "**held-out test partition**, one model per cell.",
        "",
        line(header),
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
        *(line(r) for r in rows),
        "",
        f"Model: `{oracle['kind']}` for both roles ({oracle['hidden_layers']} hidden "
        f"layer, dim {oracle['hidden_dim']}); benchmark forces ensemble size 1. The "
        "oracle trains on clean sequences and is scored clean only; the guidance "
        "predictor trains with uniform random masking and is scored at three mask "
        "levels. Checkpoints were selected on clean validation data; test labels "
        "were never used for selection or calibration.",
    ]
    (out / "spearman_table.md").write_text("\n".join(text) + "\n", encoding="utf-8")

    with (out / "spearman_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows([r[0], r[1].replace(",", ""), *r[2:]] for r in rows)


def write_png(out: Path, rows, run_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "Split",
        "Test\nvariants",
        "Oracle\n(clean)",
        "Predictor\n(clean)",
        "Predictor\n(50% masked)",
        "Predictor\n(90% masked)",
    ]
    widths = [0.17, 0.14, 0.15, 0.16, 0.19, 0.19]

    fig, ax = plt.subplots(figsize=(8.6, 3.1), dpi=220)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    table = ax.table(
        cellText=rows,
        colLabels=labels,
        colWidths=widths,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_edgecolor(RULE)
        if row == 0:
            cell.set_height(0.19)
            cell.set_text_props(color=INK, weight="bold", linespacing=1.35)
            cell.set_facecolor(ACCENT)
            cell.visible_edges = "B"
            cell.set_linewidth(1.0)
        else:
            cell.set_height(0.135)
            cell.set_facecolor("white" if row % 2 else "#f6f8fa")
            cell.set_text_props(color=INK if col >= 2 else MUTED)
        if col == 0:
            cell.set_text_props(ha="left", color=INK, weight="bold")
            cell.PAD = 0.07

    ax.set_title(
        "CreiLOV benchmark — Spearman correlation on held-out test splits",
        color=INK,
        fontsize=12,
        weight="bold",
        pad=10,
        loc="left",
        x=0,
    )
    fig.text(
        0.5,
        0.015,
        f"Run {run_name} · one-hot MLP, one model per cell · "
        "test labels unused for checkpoint selection",
        color=MUTED,
        fontsize=7.8,
        ha="center",
    )
    fig.savefig(out / "spearman_table.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="run directory holding benchmark/")
    parser.add_argument("--out", type=Path, help="output directory (default: run)")
    parser.add_argument("--name", help="run label for captions (default: directory name)")
    parser.add_argument("--no-png", action="store_true", help="skip the PNG render")
    args = parser.parse_args()

    run = args.run.resolve()
    out = (args.out or run).resolve()
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or run.name

    header, rows, config = build_rows(run)
    write_markdown(out, header, rows, config, name)
    written = ["spearman_table.md", "spearman_table.csv"]
    if not args.no_png:
        write_png(out, rows, name)
        written.append("spearman_table.png")

    print("\n".join(" | ".join(r) for r in [header, *rows]))
    print("\nwrote " + ", ".join(f"{out / f}" for f in written))


if __name__ == "__main__":
    main()
