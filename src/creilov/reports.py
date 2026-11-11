"""Local report rendering and selective Modal downloads; no model imports.

The Modal CLI environment handles downloads. Plotting runs through the project's
uv environment, so the isolated Modal CLI needs no scientific dependencies.
"""

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess

REPORT_EXTENSIONS = {".csv", ".png", ".md"}
# Run IDs may be nested, e.g. 15-positions/tag-v1, to group related runs.
RUN_ID_PATTERN = r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*"


def check_run_id(run_id: str) -> str:
    if not re.fullmatch(RUN_ID_PATTERN, run_id):
        raise ValueError(
            "run-id must be letters, digits, underscores and hyphens, optionally "
            f"nested with '/' (e.g. 15-positions/tag-v1); got {run_id!r}"
        )
    return run_id


def run_id_of(remote_dir: str) -> str:
    """Run ID of a Volume run folder: /runs/15-positions/tag-v1 is its own ID.

    The Volume mirrors outputs/, so a run ID is a path, not a flat name.
    """
    parts = PurePosixPath(remote_dir).parts
    if parts[:2] != ("/", "runs") or len(parts) < 3:
        raise ValueError(f"Not a /runs/<run-id> folder: {remote_dir}")
    return check_run_id("/".join(parts[2:]))


def volume_run_id(remote_path: str) -> str:
    """Run ID of the folder holding a Volume file, such as a library."""
    return run_id_of(PurePosixPath(remote_path).parent.as_posix())


def config_name(stage: str, suffix: str) -> str:
    """Evaluation shares a run folder with the guidance that made the library,
    so it saves its config under a distinct name rather than overwriting."""
    return f"evaluate_config.{suffix}" if stage == "evaluate" else f"config.{suffix}"


def required_inputs(stage: str, cfg: dict) -> list[str]:
    paths = [config_name(stage, "json")]
    if stage in {"splits", "benchmark", "train-predictor", "train-oracle"}:
        paths.append("split_summary.json")
    if stage == "benchmark":
        paths.append("benchmark/summary.json")
    elif stage in {"train-predictor", "train-oracle"}:
        role = stage.removeprefix("train-")
        paths.extend(["validation_summary.json", f"{role}/training.json"])
        if stage == "train-predictor" and cfg["predictor_ensemble_sizes"]:
            paths.append("predictor_ensemble_summary.json")
    elif stage == "guidance":
        paths.append("library.csv")
    elif stage == "evaluate":
        paths.extend(["metrics.json", "per_sequence.csv"])
    elif stage != "splits":
        raise ValueError(f"Unknown report stage: {stage}")
    return paths


def sync_reports(volume, stage: str, run_id: str, project_root: Path) -> Path:
    """Download only report inputs to a cache; render exports under outputs/.

    outputs/ mirrors the run Volume: a run saved at ``creilov-runs:/<stage>/<id>``
    reports to ``outputs/<stage>/<id>``. An evaluate run shares the guidance run
    folder holding the library it scored, so it reports there too.
    """
    check_run_id(run_id)
    cache = project_root / ".cache" / "report-inputs" / run_id
    cache.mkdir(parents=True, exist_ok=True)

    def download(relative: str) -> None:
        path = cache / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("wb") as handle:
            for chunk in volume.read_file(f"/{run_id}/{relative}"):
                handle.write(chunk)
        temporary.replace(path)

    settings = config_name(stage, "json")
    download(settings)
    cfg = json.loads((cache / settings).read_text(encoding="utf-8"))
    # Original benchmark runs predate run.stage; use the explicit requested stage.
    if cfg.get("stage", stage) != stage:
        raise ValueError(f"Run {run_id} contains a {cfg['stage']} config, not {stage}")
    if stage == "evaluate" and volume_run_id(cfg["library_path"]) != run_id:
        raise ValueError(
            f"Run {run_id} scored a library from another run: {cfg['library_path']}"
        )
    output = project_root / "outputs" / run_id
    for relative in required_inputs(stage, cfg):
        if relative != settings:
            download(relative)
    subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(project_root),
            "--no-sync",
            "python",
            "-m",
            "creilov.reports",
            stage,
            "--inputs",
            str(cache),
            "--out",
            str(output),
            "--run-id",
            run_id,
        ],
        cwd=project_root,
        check=True,
    )
    # Copy the launch TOML next to the reports; runs before this feature lack it.
    launch = config_name(stage, "toml")
    try:
        download(launch)
    except FileNotFoundError:
        (cache / f"{launch}.partial").unlink(missing_ok=True)
        print(f"No {launch} saved for {run_id}; skipping config copy")
    else:
        shutil.copyfile(cache / launch, output / launch)
    return output


def write_table(
    out: Path, name: str, rows: list[dict], title: str, note: str = ""
) -> None:
    """Export full precision CSV, readable Markdown and a compact table graphic."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def display(value):
        if value is None:
            return "undefined"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    cells = [[display(row.get(key)) for key in fields] for row in rows]
    markdown = (
        f"# {title}\n\n| " + " | ".join(fields) + " |\n|" + "---|" * len(fields) + "\n"
    )
    markdown += "\n".join("| " + " | ".join(row) + " |" for row in cells)
    (out / f"{name}.md").write_text(markdown + f"\n\n{note}\n", encoding="utf-8")
    fig, ax = plt.subplots(
        figsize=(max(7, len(fields) * 1.6), max(2.5, len(rows) * 0.35 + 1.6)),
        layout="constrained",
    )
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        colLabels=[f.replace("_", " ") for f in fields],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    table.auto_set_column_width(list(range(len(fields))))
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dce5")
        cell.set_facecolor(
            "#e8eef6" if r == 0 else ("#f5f7fa" if r % 2 == 0 else "white")
        )
    ax.set_title(title, fontsize=15, pad=18)
    if note:
        fig.supxlabel(note, fontsize=8)
    fig.savefig(out / f"{name}.png", dpi=180)
    plt.close(fig)


def performance_rows(stage: str, source: Path, cfg: dict) -> list[dict]:
    if stage == "benchmark":
        entries = json.loads((source / "benchmark/summary.json").read_text())
    else:
        role = stage.removeprefix("train-")
        if role == "predictor" and cfg["predictor_ensemble_sizes"]:
            summary = json.loads(
                (source / "predictor_ensemble_summary.json").read_text()
            )
            entries = [
                {"role": role, "ensemble_size": int(size), **r}
                for size, entry in summary.items()
                for r in entry["reports"]
            ]
        else:
            summary = json.loads((source / "validation_summary.json").read_text())
            entries = [
                {"role": role, "ensemble_size": cfg[role]["ensemble_size"], **r}
                for r in summary[role]
            ]
    fields = (
        "split",
        "role",
        "ensemble_size",
        "partition",
        "mask_fraction",
        "n",
        "spearman",
        "rmse",
        "mae",
        "brier",
        "gaussian_nll",
        "coverage_95",
    )
    return [{key: entry[key] for key in fields if key in entry} for entry in entries]


def plot_training(source: Path, out: Path, role: str) -> None:
    import matplotlib.pyplot as plt

    training = json.loads((source / role / "training.json").read_text())
    histories = training["history"]
    with (out / "learning_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["member", "epoch", "train_standardized_mse", "validation_mse"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"member": member + 1, **epoch}
                         for member, history in enumerate(histories) for epoch in history)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for member, history in enumerate(histories):
        epochs = history
        ax.plot(
            [e["epoch"] for e in epochs],
            [e["validation_mse"] for e in epochs],
            label=f"Member {member + 1}",
            alpha=0.75,
        )
    ax.set(
        xlabel="Epoch",
        ylabel="Clean validation MSE",
        title=f"{role.capitalize()} learning curves",
    )
    if len(histories) <= 5:
        ax.legend()
    fig.savefig(out / "learning_curves.png", dpi=180)
    plt.close(fig)


def plot_evaluation(
    out: Path, oracle, predictor, mutations, stats: dict, bins: int, run_id: str
) -> None:
    """Evaluation figures, titled with the guidance run that made the library."""
    import matplotlib.pyplot as plt
    import numpy as np

    oracle, predictor, mutations = map(np.asarray, (oracle, predictor, mutations))
    mutation_bins = np.arange(mutations.min() - 0.5, mutations.max() + 1.5)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    axes[0, 0].hist(oracle, bins=bins, color="#4477AA", edgecolor="white")
    axes[0, 0].set(
        xlabel="Oracle score (log10 fluorescence)",
        ylabel="Sequences",
        title="Library oracle scores",
    )
    axes[0, 1].hist(mutations, bins=mutation_bins, color="#228833", edgecolor="white")
    axes[0, 1].set(
        xlabel="Mutations from wild type",
        ylabel="Sequences",
        title="Mutational distance",
    )
    axes[1, 0].scatter(predictor, oracle, s=15, alpha=0.6, color="#AA3377")
    axes[1, 0].set(
        xlabel="Clean predictor score",
        ylabel="Oracle score",
        title="Predictor–oracle agreement",
    )
    axes[1, 1].axis("off")
    labels = {
        "diversity_mean_pairwise_hamming_unique": "Diversity (unique pair Hamming)"
    }
    table = axes[1, 1].table(
        cellText=[
            (
                labels.get(k, k.replace("_", " ")),
                "undefined" if v is None else f"{v:.5g}",
            )
            for k, v in stats.items()
        ],
        colWidths=[0.77, 0.23],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.65)
    fig.suptitle(f"CreiLOV library evaluation: {run_id}", fontsize=17)
    fig.savefig(out / "evaluation_summary.png", dpi=180)
    plt.close(fig)
    # Standalone histograms are convenient to reuse in reports.
    for values, name, xlabel, edges in (
        (oracle, "oracle_scores", "Oracle score (log10 fluorescence)", bins),
        (mutations, "mutations", "Mutations from wild type", mutation_bins),
    ):
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        ax.hist(values, bins=edges, color="#4477AA", edgecolor="white")
        ax.set(xlabel=xlabel, ylabel="Sequences", title=run_id)
        fig.savefig(out / f"{name}_histogram.png", dpi=180)
        plt.close(fig)


def render_reports(stage: str, source: Path, out: Path, run_id: str = "") -> None:
    """Build reports from cached summaries, without loading a model or dataset.

    ``run_id`` is the run's path on the Volume below ``/runs``; it titles the
    figures, so a PNG on its own still names the run that produced it.
    """
    out.mkdir(parents=True, exist_ok=True)
    run_id = run_id or out.name
    cfg = json.loads(
        (source / config_name(stage, "json")).read_text(encoding="utf-8")
    )
    import matplotlib

    matplotlib.use("Agg")
    note = ""
    if stage in {"splits", "benchmark", "train-predictor", "train-oracle"}:
        sizes = json.loads((source / "split_summary.json").read_text())
        write_table(
            out,
            "split_sizes",
            [{"split": name, **counts} for name, counts in sizes.items()],
            "CreiLOV data splits",
        )
    if stage in {"benchmark", "train-predictor", "train-oracle"}:
        rows = performance_rows(stage, source, cfg)
        # Keep all metrics in CSV; use compact columns in the screenshots.
        with (out / "performance_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        compact = [
            {
                k: v
                for k, v in r.items()
                if k
                in {
                    "split",
                    "role",
                    "ensemble_size",
                    "mask_fraction",
                    "n",
                    "spearman",
                    "rmse",
                }
            }
            for r in rows
        ]
        note = (
            "Held-out test metrics; one model per role and split."
            if stage == "benchmark"
            else "Validation diagnostics used for checkpoint selection/calibration; test data remain untouched."
        )
        write_table(
            out, "performance_summary", compact, f"CreiLOV {stage} performance", note
        )
        if stage == "benchmark":
            from .benchmark_report import build_rows, write_markdown, write_png

            header, table_rows, config = build_rows(source)
            write_markdown(out, header, table_rows, config, run_id)
            write_png(out, table_rows, run_id)
        else:
            # Reuse the existing detailed ensemble plots for the standard levels.
            if (
                stage == "train-predictor"
                and cfg["predictor_ensemble_sizes"]
                and set(cfg["evaluation"]["mask_fractions"]) == {0.0, 0.5, 0.9}
            ):
                from .predictor_report import build_table

                build_table(source)
                for name in (
                    "ensemble_metrics.csv",
                    "ensemble_comparison.md",
                    "ensemble_comparison.png",
                    "ensemble_performance.png",
                ):
                    shutil.copyfile(source / name, out / name)
            plot_training(source, out, stage.removeprefix("train-"))
    elif stage == "guidance":
        with (source / "library.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("Cannot report an empty library")
        # A run that stopped early wrote fewer draws than it was asked for.
        if len(rows) != cfg["guidance"]["num_samples"]:
            raise ValueError(
                f"Generation run is incomplete: {len(rows)} of "
                f"{cfg['guidance']['num_samples']} draws; no library report available"
            )
        shutil.copyfile(source / "library.csv", out / "library.csv")
        note = (
            "Duplicate draws are retained. Scores, mutation counts and diversity "
            "are added to this folder by the evaluate stage."
        )
    elif stage == "evaluate":
        stats = json.loads((source / "metrics.json").read_text())
        shutil.copyfile(source / "per_sequence.csv", out / "per_sequence.csv")
        with (source / "per_sequence.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        plot_evaluation(
            out,
            [float(r["oracle_score"]) for r in rows],
            [float(r["predictor_score"]) for r in rows],
            [int(r["n_mutations"]) for r in rows],
            {k: v for k, v in stats.items() if not isinstance(v, dict)},
            cfg["histogram_bins"],
            run_id,
        )
        write_table(
            out,
            "evaluation_metrics",
            [
                {"metric": k, "value": v}
                for k, v in stats.items()
                if not isinstance(v, dict)
            ],
            f"CreiLOV library evaluation: {run_id}",
        )
        note = "\n".join(
            f"- **{key}**: {value}" for key, value in stats["definitions"].items()
        )
    elif stage != "splits":
        raise ValueError(f"Unknown report stage: {stage}")
    # Include the full current report inventory, including regenerated files.
    reports = sorted(
        p.name
        for p in out.iterdir()
        if p.suffix in REPORT_EXTENSIONS | {".toml"} and p.name != "README.md"
    )
    title = run_id.rpartition("/")[2]
    text = (
        f"# CreiLOV {stage}: {title}\n\nSource: `creilov-runs:/{run_id}`. "
        "Checkpoints and raw arrays remain on Modal.\n\n"
    )
    text += "\n".join(f"- [{name}]({name})" for name in reports) + f"\n\n{note}\n"
    (out / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default="", help="Modal run behind --inputs")
    args = parser.parse_args()
    render_reports(args.stage, args.inputs, args.out, args.run_id)
    print(f"Local reports: {args.out}")


if __name__ == "__main__":
    main()
