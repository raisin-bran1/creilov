"""Small local report fixtures; no training, models or remote compute."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from creilov.reports import render_reports, sync_reports, REPORT_EXTENSIONS
from creilov.data import CREILOV_WT


class FakeVolume:
    def __init__(self, files):
        self.files = files
        self.reads = []

    def read_file(self, path):
        self.reads.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]


class ReportTests(unittest.TestCase):
    def test_selective_download_and_uv_render_do_not_fetch_models_or_arrays(self):
        cfg = {"stage": "train-oracle"}
        names = (
            "config.json",
            "split_summary.json",
            "validation_summary.json",
            "oracle/training.json",
        )
        run_id = "train-oracle/oracle"
        files = {
            f"/{run_id}/" + name: json.dumps(
                cfg if name == "config.json" else {}
            ).encode()
            for name in names
        }
        files[f"/{run_id}/config.toml"] = b'[run]\nstage = "train-oracle"\n'
        volume = FakeVolume(files)
        with tempfile.TemporaryDirectory() as directory, patch(
            "creilov.reports.subprocess.run"
        ) as run:
            root = Path(directory)
            # The patched renderer creates nothing; the real one makes outputs/.
            run.side_effect = lambda command, **_: Path(
                command[command.index("--out") + 1]
            ).mkdir(parents=True, exist_ok=True)
            out = sync_reports(volume, "train-oracle", run_id, root)
            self.assertEqual(out, root / "outputs" / run_id)
            self.assertEqual(set(volume.reads), set(files))
            self.assertEqual(run.call_args.args[0][:2], ["uv", "run"])
            self.assertIn("creilov.reports", run.call_args.args[0])
            self.assertFalse(list(out.rglob("*.json")))
            self.assertEqual(
                (out / "config.toml").read_bytes(), files[f"/{run_id}/config.toml"]
            )
            # Runs saved before the TOML copy existed still sync without it.
            del files[f"/{run_id}/config.toml"]
            with patch("builtins.print"):
                old = sync_reports(volume, "train-oracle", run_id, root / "old")
            self.assertFalse((old / "config.toml").exists())
            cache = root / "old/.cache/report-inputs" / run_id
            self.assertFalse(list(cache.glob("*.partial")))
        for bad in ("../outside", "/abs", "a//b", "a/"):
            with self.assertRaisesRegex(ValueError, "run-id"):
                sync_reports(volume, "train-oracle", bad, Path("."))

    def test_evaluate_reports_join_their_nested_guidance_folder(self):
        run_id = "guidance/15-positions/tag-v1"
        cfg = {
            "stage": "evaluate",
            "library_path": f"/runs/{run_id}/library.csv",
        }
        files = {
            f"/{run_id}/evaluate_config.json": json.dumps(cfg).encode(),
            f"/{run_id}/metrics.json": b"{}",
            f"/{run_id}/per_sequence.csv": b"id\n",
            f"/{run_id}/evaluate_config.toml": b'[run]\nstage = "evaluate"\n',
        }
        volume = FakeVolume(files)
        with tempfile.TemporaryDirectory() as directory, patch(
            "creilov.reports.subprocess.run"
        ) as run:
            root = Path(directory)
            run.side_effect = lambda command, **_: Path(
                command[command.index("--out") + 1]
            ).mkdir(parents=True, exist_ok=True)
            out = sync_reports(volume, "evaluate", run_id, root)
            self.assertEqual(out, root / "outputs" / run_id)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--run-id") + 1], run_id)
            # The guidance config.toml in that folder must survive.
            self.assertTrue((out / "evaluate_config.toml").is_file())
            self.assertFalse((out / "config.toml").exists())
            other = dict(files)
            other[f"/{run_id}/evaluate_config.json"] = json.dumps(
                {**cfg, "library_path": "/runs/guidance/elsewhere/library.csv"}
            ).encode()
            with self.assertRaisesRegex(ValueError, "another run"):
                sync_reports(FakeVolume(other), "evaluate", run_id, root)

    def test_render_each_saved_stage_from_small_fixtures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "inputs"
            source.mkdir()

            def write(name, value):
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")

            cfg = {
                "histogram_bins": 5,
                "library_path": "/runs/guidance/15-positions/tag-v1/library.csv",
                "guidance": {"num_samples": 2},
                "device": "cuda",
                "seed": 42,
                "csv_path": "/data/creilov_combined.csv",
                "predictor_ensemble_sizes": [1],
                "oracle": {
                    "kind": "one_hot_mlp",
                    "hidden_layers": 1,
                    "hidden_dim": 400,
                    "ensemble_size": 1,
                },
                "predictor": {
                    "kind": "one_hot_mlp",
                    "hidden_layers": 1,
                    "hidden_dim": 400,
                    "ensemble_size": 1,
                },
                "evaluation": {"mask_fractions": [0.0, 0.5, 0.9]},
            }
            write("config.json", cfg)
            write("evaluate_config.json", {**cfg, "stage": "evaluate"})
            write(
                "split_summary.json",
                {
                    s: {"train": 8, "validation": 1, "test": 1}
                    for s in ("random", "distance", "activity", "position")
                },
            )
            clean = {
                "partition": "validation",
                "mask_fraction": 0.0,
                "n": 3,
                "spearman": 0.9,
                "rmse": 0.1,
                "mae": 0.08,
                "brier": 0.1,
                "gaussian_nll": 0.1,
                "coverage_95": 1.0,
                "used_for_checkpoint_selection_and_calibration": True,
            }
            reports = [{**clean, "mask_fraction": mask} for mask in (0.0, 0.5, 0.9)]
            write("validation_summary.json", {"oracle": [clean], "predictor": reports})
            training = {
                "train_indices": [0, 1],
                "history": [
                    [
                        {"epoch": 1, "validation_mse": 0.2},
                        {"epoch": 2, "validation_mse": 0.1},
                    ]
                ],
            }
            write("oracle/training.json", training)
            write("predictor/training.json", training)
            write("predictor_ensemble_summary.json", {"1": {"reports": reports}})
            write(
                "benchmark/summary.json",
                [
                    {
                        **r,
                        "partition": "test",
                        "split": s,
                        "role": role,
                        "ensemble_size": 1,
                        "used_for_checkpoint_selection_and_calibration": False,
                    }
                    for s in ("random", "distance", "activity", "position")
                    for role in ("oracle", "predictor")
                    for r in ([clean] if role == "oracle" else reports)
                ],
            )
            with (source / "library.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "sequence", "n_mutations"])
                writer.writerows(
                    [("wt", CREILOV_WT, 0), ("mut", "C" + CREILOV_WT[1:], 1)]
                )
            write(
                "metrics.json",
                {
                    "oracle_score_mean": 4.1,
                    "oracle_score_variance": 0.1,
                    "predictor_oracle_pearson": None,
                    "definitions": {"variance": "population"},
                },
            )
            (source / "per_sequence.csv").write_text(
                "id,sequence,oracle_score,predictor_score,n_mutations\n"
                "a,S,4.1,4.0,0\nb,S,4.3,4.2,2\n"
            )
            for stage in (
                "splits",
                "benchmark",
                "train-oracle",
                "train-predictor",
                "guidance",
                "evaluate",
            ):
                with self.subTest(stage=stage):
                    out = root / stage
                    render_reports(stage, source, out)
                    self.assertTrue((out / "README.md").is_file())
                    self.assertTrue(list(out.glob("*.csv")))
                    if stage != "guidance":
                        self.assertTrue(list(out.glob("*.png")))
                    self.assertTrue(
                        all(p.suffix in REPORT_EXTENSIONS for p in out.iterdir())
                    )
            # Guidance exports only the library; evaluate adds scored reports.
            self.assertEqual(
                {p.name for p in (root / "guidance").iterdir()},
                {"library.csv", "README.md"},
            )
            render_reports(
                "evaluate", source, root / "guidance", "guidance/15-positions/tag-v1"
            )
            self.assertTrue((root / "guidance/evaluation_summary.png").is_file())
            readme = (root / "guidance/README.md").read_text()
            self.assertIn("# CreiLOV evaluate: tag-v1", readme)
            self.assertIn("creilov-runs:/guidance/15-positions/tag-v1", readme)
            # One row short of the requested draws means the run stopped early.
            write("config.json", {**cfg, "guidance": {"num_samples": 3}})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                render_reports("guidance", source, root / "partial")


if __name__ == "__main__":
    unittest.main()
