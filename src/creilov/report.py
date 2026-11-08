"""Create shareable HTML and screenshot-ready PNG tables from a benchmark run."""

import argparse
import csv
from html import escape
import json
from pathlib import Path

SPLITS = ("random", "distance", "activity", "position")
ROLES = ("oracle", "predictor")
SPLIT_LABELS = {
    "random": "Random",
    "distance": "Distance",
    "activity": "Activity",
    "position": "Position",
}
ROLE_LABELS = {
    "oracle": "Oracle (clean training)",
    "predictor": "Predictor (masked training)",
}


def number(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def png_table(
    path: Path,
    title: str,
    subtitle: str,
    headers: list[str],
    rows: list[list[str]],
    footnote: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, max(6.0, 3.1 + len(rows) * 0.43)), dpi=160)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    fig.subplots_adjust(left=0.035, right=0.965, top=0.83, bottom=0.13)
    fig.text(
        0.035, 0.95, title, fontsize=23, fontweight="bold", color="#172b45", va="top"
    )
    fig.text(0.035, 0.875, subtitle, fontsize=12, color="#526278", va="top")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="right",
        colLoc="right",
        loc="center",
        bbox=[0, 0.04, 1, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor("#dce3eb")
        cell.PAD = 0.10
        if row == 0:
            cell.set_facecolor("#172b45")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f0f5fa" if (row - 1) // 2 % 2 == 0 else "white")
        if col < 2:
            cell.set_text_props(ha="left")
    table.auto_set_column_width(list(range(len(headers))))
    fig.text(
        0.035, 0.075, footnote, fontsize=10, color="#526278", va="top", linespacing=1.5
    )
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def build_report(run_dir: Path, output_dir: Path) -> Path:
    summary = json.loads(
        (run_dir / "benchmark/summary.json").read_text(encoding="utf-8")
    )
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    sizes = json.loads((run_dir / "split_summary.json").read_text(encoding="utf-8"))
    clean = {(r["split"], r["role"]): r for r in summary if r["mask_fraction"] == 0}
    if set(clean) != {(s, r) for s in SPLITS for r in ROLES}:
        raise ValueError("Report requires all eight completed experiments")
    if any(r["ensemble_size"] != 1 or r["partition"] != "test" for r in summary):
        raise ValueError("Expected single-member held-out test benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_headers = [
        "Split",
        "Model role",
        "Training n",
        "Test n",
        "Best epoch",
        "RMSE ↓",
        "MAE ↓",
        "Spearman ρ ↑",
    ]
    clean_rows, calibration_rows = [], []
    predictor_budgets = set()
    for split in SPLITS:
        for role in ROLES:
            row = clean[(split, role)]
            metadata = json.loads(
                (run_dir / "benchmark" / split / role / "training.json").read_text(
                    encoding="utf-8"
                )
            )
            history = metadata["history"][0]
            best_epoch = min(history, key=lambda h: h["validation_mse"])["epoch"]
            if role == "predictor":
                predictor_budgets.add(len(metadata["train_indices"]))
            clean_rows.append(
                [
                    SPLIT_LABELS[split],
                    ROLE_LABELS[role],
                    f"{len(metadata['train_indices']):,}",
                    f"{row['n']:,}",
                    f"{best_epoch} / {len(history)}",
                    number(row["rmse"]),
                    number(row["mae"]),
                    number(row["spearman"]),
                ]
            )
            calibration_rows.append(
                [
                    SPLIT_LABELS[split],
                    ROLE_LABELS[role],
                    number(row["brier"], 4),
                    number(row["gaussian_nll"]),
                    f"{row['coverage_95']:.1%}",
                    number(row["top_k_precision"]),
                    number(row["top_k_enrichment"]),
                ]
            )
    masked_headers = [
        "Split",
        "Masked inputs",
        "Test n",
        "RMSE ↓",
        "MAE ↓",
        "Spearman ρ ↑",
    ]
    masked_rows = []
    for split in SPLITS:
        for row in sorted(
            [
                r
                for r in summary
                if r["split"] == split
                and r["role"] == "predictor"
                and r["mask_fraction"] > 0
            ],
            key=lambda r: r["mask_fraction"],
        ):
            masked_rows.append(
                [
                    SPLIT_LABELS[split],
                    f"{row['mask_fraction']:.0%}",
                    f"{row['n']:,}",
                    number(row["rmse"]),
                    number(row["mae"]),
                    number(row["spearman"]),
                ]
            )
    calibration_headers = [
        "Split",
        "Model role",
        "Brier ↓",
        "Gaussian NLL ↓",
        "95% coverage",
        "Precision@100",
        "Enrichment@100",
    ]
    top_k = config["evaluation"]["top_k"]
    calibration_headers[-2:] = [f"Precision@{top_k}", f"Enrichment@{top_k}"]
    split_cfg = config["splits"]
    design_headers = [
        "Split",
        "Training pool",
        "Validation",
        "Test",
        "Holdout definition",
    ]
    definitions = {
        "random": f"Random {split_cfg['train_fraction']:.0%} / {split_cfg['validation_fraction']:.0%} / {1-split_cfg['train_fraction']-split_cfg['validation_fraction']:.0%}",
        "distance": f"Train ≤{split_cfg['distance_train_max']}; validation {split_cfg['distance_train_max']+1}–{split_cfg['distance_validation_max']}; test ≥{split_cfg['distance_validation_max']+1}",
        "activity": f"Train bottom {split_cfg['activity_train_quantile']:.0%}; test top {1-split_cfg['activity_validation_quantile']:.0%}",
        "position": f"Validation site {split_cfg['validation_positions']}; test site {split_cfg['test_positions']}",
    }
    design_rows = [
        [
            SPLIT_LABELS[s],
            f"{sizes[s]['train']:,}",
            f"{sizes[s]['validation']:,}",
            f"{sizes[s]['test']:,}",
            definitions[s],
        ]
        for s in SPLITS
    ]
    seed = config["seed"]
    names = {
        "one_hot_mlp": "One-hot MLP",
        "linear_probe": "Frozen ESM probe",
        "additive": "Additive model",
    }
    kinds = list(dict.fromkeys(config[r]["kind"] for r in ROLES))
    architecture = " / ".join(names.get(k, k) for k in kinds)
    subtitle = f"CreiLOV S2 · {architecture} · seed {seed} · one model per experiment · held-out test data"
    budgets = ", ".join(f"{n:,}" for n in sorted(predictor_budgets))
    note = f"Errors are in log10 fluorescence units. Best epoch / total epochs shown. No confidence intervals: one seed, one member.\nPredictor gradient-training rows: {budgets}, plus separate validation labels. Oracle uses its full training pool."
    tables = [
        (
            "split_design",
            "Benchmark data partitions",
            design_headers,
            design_rows,
            "Positions use 1-based sequence numbering; the test-site rule takes precedence. Other background mutations are allowed.\nTraining pools and validation pools are disjoint from test data. Predictor training is subsampled from its training pool.",
        ),
        (
            "clean_performance",
            "Clean-sequence prediction",
            clean_headers,
            clean_rows,
            note,
        ),
        (
            "masked_performance",
            "Masked-input guidance predictor",
            masked_headers,
            masked_rows,
            "Same four noisy-trained predictors, evaluated at additional masking levels; these are not additional trained models.\nMasks are sampled independently on residue positions. These measurements assess prediction, not TAG approximation quality.",
        ),
        (
            "calibration",
            "Threshold probability and candidate ranking",
            calibration_headers,
            calibration_rows,
            f"Event: measured log10 fluorescence > {config['guidance']['threshold']}. Coverage target: 95%. Single-model variance comes from validation residuals.\nNo positive labels in a test partition yields undefined enrichment (—). These are assay predictions, not wet-lab generated-protein results.",
        ),
    ]
    sections = []
    for stem, title, headers, rows, footnote in tables:
        if not rows:
            continue
        png_table(output_dir / f"{stem}.png", title, subtitle, headers, rows, footnote)
        with (output_dir / f"{stem}.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
        sections.append(
            f"<section><h2>{escape(title)}</h2>{html_table(headers, rows)}<p class='note'>{escape(footnote).replace(chr(10), '<br>')}</p><a href='{stem}.png'>Download screenshot table</a></section>"
        )
    html = (
        """<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>CreiLOV S2 benchmark results</title><style>
body{font:16px system-ui,sans-serif;color:#172b45;background:#edf2f7;margin:0;padding:36px}main{max-width:1400px;margin:auto}
header,section{background:white;padding:28px 32px;margin-bottom:24px;border-radius:12px}h1{font-size:32px;margin:0 0 12px}h2{margin:0 0 22px}
table{border-collapse:collapse;width:100%;font-size:15px}th{background:#172b45;color:white}th,td{padding:14px 12px;text-align:right;border-bottom:1px solid #dce3eb}
th:nth-child(-n+2),td:nth-child(-n+2){text-align:left}tbody tr:nth-child(4n+1),tbody tr:nth-child(4n+2){background:#f0f5fa}
.note{color:#526278;font-size:14px;line-height:1.6}a{color:#2366a2}@media print{body{padding:0;background:white}section{break-inside:avoid}a{display:none}}
</style><main><header><h1>CreiLOV S2 benchmark results</h1>"""
        + f"<p>{escape(subtitle)}</p><p class='note'>Architecture and optimizer settings follow the saved run configuration. Benchmark overrides ensemble size to one. Validation selects checkpoints and estimates residual variance; reported metrics use disjoint test sets. Oracle and guidance predictor differ in data budget and masking, so this is not a controlled architecture comparison.</p></header>"
    )
    html += "".join(sections) + "</main></html>"
    path = output_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# CreiLOV S2 benchmark report\n\nOpen index.html for all tables. PNG files are screenshot-ready; CSV files contain the displayed values.\n\n"
        f"Seed: {seed}. Eight single-member experiments, not eight ensembles. Raw inputs are in {run_dir.resolve()}.\n"
        "Validation controls selection/calibration; test sets supply benchmark metrics. No repeated-seed uncertainty estimates are available.\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(build_report(args.run_dir, args.output_dir))


if __name__ == "__main__":
    main()
