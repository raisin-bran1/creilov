"""Stage isolation, shared benchmark settings, and portable Modal configs."""

from dataclasses import replace
import ast
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import re
import sys

from creilov.config import (
    DEFAULT_CONFIGS,
    default_run_id,
    deserialize_config,
    qualify_run_id,
    stage_folder,
    load_config,
    modal_config,
    serialize_config,
    validate_stage_config,
    guidance_overrides,
)

ROOT = Path(__file__).parents[1]


def guided_config():
    """The shipped guidance config with the control arm explicitly off.

    ``guidance.control`` is an experiment switch that gets flipped between
    runs, and a control run deliberately carries no predictor checkpoint. Tests
    covering the guided arm's contract pin the flag instead of inheriting
    whichever arm the config currently selects.
    """
    text = re.sub(
        r"(?m)^control\s*=.*$",
        "control = false",
        (ROOT / DEFAULT_CONFIGS["guidance"]).read_text(encoding="utf-8"),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "guidance.toml"
        path.write_text(text, encoding="utf-8")
        return load_config(path)  # Fully materialized; the temp file can go.


class ConfigTests(unittest.TestCase):
    def test_nested_output_dir_becomes_a_nested_run_id_on_the_volume(self):
        """The Volume mirrors outputs/, so a TOML's output_dir places the run."""
        cfg = guided_config()
        nested = replace(cfg, output_dir=Path("/runs/guidance/15-positions/tag-v1"))
        self.assertEqual(default_run_id(nested), "guidance/15-positions/tag-v1")
        remote = modal_config(nested, default_run_id(nested), "")
        self.assertEqual(remote.output_dir, nested.output_dir)
        # Every shipped config already names its stage folder.
        for stage, name in DEFAULT_CONFIGS.items():
            with self.subTest(stage=stage):
                run_id = default_run_id(load_config(ROOT / name))
                # splits shares the benchmark config; evaluate joins guidance.
                folder = {"splits": "benchmark", "evaluate": "guidance"}
                self.assertEqual(
                    run_id.partition("/")[0], folder.get(stage, stage)
                )
        for bad in ("/data/elsewhere", "/runs", "relative/path"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    default_run_id(replace(cfg, output_dir=Path(bad)))

    def test_run_id_resolves_against_the_stage_folder(self):
        """--run-id names the nest; the stage folder it belongs in is implied."""
        cfg = guided_config()
        # A bare nest lands under the stage folder, not at the Volume root.
        self.assertEqual(
            qualify_run_id(cfg, "15-positions/tag-v1"),
            "guidance/15-positions/tag-v1",
        )
        self.assertEqual(qualify_run_id(cfg, "pilot"), "guidance/pilot")
        # Already-qualified IDs are untouched, so existing invocations still work.
        for already in ("guidance", "guidance/pilot", "guidance/15-positions/x"):
            with self.subTest(already=already):
                self.assertEqual(qualify_run_id(cfg, already), already)
        # "guidance" must match a whole segment: guidance-extra is its own nest.
        self.assertEqual(
            qualify_run_id(cfg, "guidance-extra"), "guidance/guidance-extra"
        )
        # An empty run ID means "use the TOML's own output_dir".
        self.assertEqual(qualify_run_id(cfg, ""), "")
        # Off-Volume configs have no stage folder to borrow, so they pass through.
        outside = replace(cfg, output_dir=Path("/data/elsewhere"))
        self.assertEqual(stage_folder(outside), "")
        self.assertEqual(qualify_run_id(outside, "pilot"), "pilot")

    def test_stage_folder_follows_the_config_not_the_stage_name(self):
        """splits shares the benchmark config; evaluate joins the guidance run."""
        # A --stage splits launch loads benchmark.toml, whose own stage is
        # "benchmark", so the folder follows the config and needs no special case.
        splits = load_config(ROOT / DEFAULT_CONFIGS["splits"])
        self.assertEqual(splits.stage, "benchmark")
        self.assertEqual(stage_folder(splits), "benchmark")
        self.assertEqual(qualify_run_id(splits, "pilot"), "benchmark/pilot")
        evaluate = load_config(ROOT / DEFAULT_CONFIGS["evaluate"])
        library = "/runs/guidance/15-positions/control/library.fasta"
        self.assertEqual(stage_folder(evaluate, library), "guidance")
        self.assertEqual(
            qualify_run_id(evaluate, "15-positions/control", library),
            "guidance/15-positions/control",
        )

    def test_modal_entrypoint_uses_toml_inputs_and_output_without_overrides(self):
        module = ast.parse((ROOT / "modal_app.py").read_text())
        function = next(
            n
            for n in module.body
            if isinstance(n, ast.FunctionDef) and n.name == "main"
        )
        function.decorator_list = []
        remote = Mock(return_value={"ok": True})
        namespace = {
            "ROOT": ROOT,
            "sys": sys,
            "re": re,
            "GPU": "T4",
            "TIMEOUT": 1800,
            "run_job": SimpleNamespace(remote=remote),
            "output_volume": Mock(),
        }
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]), "modal_app.py", "exec"
            ),
            namespace,
        )
        # Evaluation has no run of its own: it joins the run holding the
        # library, so its run ID is that library's folder below /runs.
        expected_run = load_config(
            ROOT / DEFAULT_CONFIGS["evaluate"]
        ).library_path.parent.as_posix().removeprefix("/runs/")
        with patch("builtins.print"), patch("creilov.reports.sync_reports") as sync:
            namespace["main"](stage="evaluate")
            sync.assert_called_once_with(
                namespace["output_volume"], "evaluate", expected_run, ROOT
            )
        stage, payload, run_id, toml_text = remote.call_args.args
        self.assertEqual(
            toml_text,
            (ROOT / DEFAULT_CONFIGS["evaluate"]).read_text(encoding="utf-8"),
        )
        self.assertEqual(stage, "evaluate")
        self.assertEqual(run_id, expected_run)
        self.assertIn("/", run_id)  # nested, mirroring outputs/
        cfg = json.loads(payload)
        # Structural, not literal: these paths are re-pointed between runs, and
        # the entrypoint's job is to forward Volume paths, not specific run IDs.
        self.assertRegex(cfg["library_path"], r"^/runs/.+/library\.(fasta|csv)$")
        self.assertRegex(cfg["predictor_checkpoint"], r"^/runs/.+/model\.pt$")
        self.assertRegex(cfg["oracle_checkpoint"], r"^/runs/.+/model\.pt$")
        self.assertEqual(cfg["output_dir"], f"/runs/{expected_run}")
        self.assertEqual(
            cfg["library_path"].rpartition("/")[0], cfg["output_dir"]
        )
        remote.reset_mock()
        with patch("builtins.print"), patch("creilov.reports.sync_reports") as sync:
            namespace["main"](stage="evaluate", reports_only=True)
            remote.assert_not_called()
            sync.assert_called_once()

    def test_generation_overrides_validate_and_preserve_other_settings(self):
        cfg = load_config(ROOT / DEFAULT_CONFIGS["guidance"])
        result = guidance_overrides(cfg, 7, "deg")
        self.assertEqual(result.guidance.num_samples, 7)
        self.assertEqual(result.guidance.method, "deg")
        self.assertEqual(result.predictor_checkpoint, cfg.predictor_checkpoint)
        self.assertEqual(result, deserialize_config(serialize_config(result)))
        for n, method in ((0, "tag"), (-2, "tag"), (4, "invalid")):
            with self.subTest(n=n, method=method), self.assertRaises(ValueError):
                guidance_overrides(cfg, n, method)
        with self.assertRaisesRegex(ValueError, "only used by guidance"):
            guidance_overrides(
                load_config(ROOT / DEFAULT_CONFIGS["train-predictor"]), 5, "tag"
            )

    def test_stages_have_only_their_models_and_distinct_outputs(self):
        configs = {
            stage: load_config(ROOT / filename)
            for stage, filename in DEFAULT_CONFIGS.items()
            if stage != "splits"
        }
        configs["guidance"] = guided_config()
        predictor, oracle = configs["train-predictor"], configs["train-oracle"]
        self.assertIsNone(predictor.oracle)
        self.assertIsNone(oracle.predictor)
        self.assertEqual(predictor.predictor_ensemble_sizes, [1, 5, 20])
        self.assertEqual(len({c.output_dir for c in configs.values()}), 5)
        # Guidance must consume a predictor checkpoint from the Volume; which
        # run or ensemble size it points at is a tuning choice, not a contract.
        self.assertRegex(
            configs["guidance"].predictor_checkpoint.as_posix(),
            r"^/runs/.+/predictor/.*model\.pt$",
        )
        for stage, cfg in configs.items():
            validate_stage_config(stage, cfg)
            self.assertEqual(cfg, deserialize_config(serialize_config(cfg)))
            self.assertTrue(cfg.csv_path.as_posix().startswith("/data/"))
            self.assertTrue(cfg.output_dir.as_posix().startswith("/runs/"))
            self.assertEqual(cfg.device, "cuda")
        with self.assertRaisesRegex(ValueError, "cannot use"):
            validate_stage_config("train-oracle", predictor)

    def test_benchmark_reads_current_role_settings_but_owns_its_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in (ROOT / "configs").glob("*.toml"):
                shutil.copyfile(path, root / path.name)
            path = root / "train_predictor.toml"
            text = (
                path.read_text()
                .replace("hidden_dim = 400", "hidden_dim = 512")
                .replace("learning_rate = 0.001", "learning_rate = 0.003")
                .replace("distance_train_max = 7", "distance_train_max = 6")
            )
            path.write_text(text)
            cfg = load_config(root / "benchmark.toml")
            self.assertEqual(cfg.predictor.hidden_dim, 512)
            self.assertEqual(cfg.predictor_training.learning_rate, 0.003)
            self.assertEqual(cfg.splits.distance_train_max, 7)
            self.assertEqual(cfg.predictor_ensemble_sizes, [])
            self.assertEqual(cfg.csv_path.name, "creilov.csv")

    def test_guidance_control_drops_the_predictor_checkpoint(self):
        guide = guided_config()
        self.assertIsNotNone(guide.predictor_checkpoint)
        # Either route into the control arm, from the TOML or the CLI, must
        # leave the saved config with no predictor to credit.
        for control in (
            replace(guide, guidance=replace(guide.guidance, control=True)),
            guidance_overrides(guide, None, None, control=True),
        ):
            self.assertTrue(control.guidance.control)
            self.assertIsNone(control.predictor_checkpoint)
            validate_stage_config("guidance", control)
            self.assertEqual(control, deserialize_config(serialize_config(control)))
            remote = modal_config(control, "guide-control-v1", "")
            self.assertIsNone(remote.predictor_checkpoint)
            self.assertEqual(remote.output_dir.as_posix(), "/runs/guide-control-v1")
            with self.assertRaisesRegex(ValueError, "unused by a control run"):
                modal_config(control, "guide-control-v1", "/runs/p/predictor/model.pt")
        with self.assertRaisesRegex(ValueError, "requires run.predictor_checkpoint"):
            replace(guide, predictor_checkpoint=None)
        evaluate = load_config(ROOT / DEFAULT_CONFIGS["evaluate"])
        with self.assertRaisesRegex(ValueError, "only applies to the guidance stage"):
            replace(evaluate, guidance=replace(evaluate.guidance, control=True))
        with self.assertRaisesRegex(ValueError, "only used by guidance"):
            guidance_overrides(evaluate, None, None, control=True)

    def test_mask_positions_are_validated_and_exclusive_with_mask_fraction(self):
        guide = guided_config()
        # The shipped config designs the 15 sites varied by the si_002 library.
        self.assertEqual(
            guide.guidance.mask_positions,
            [3, 4, 5, 7, 29, 34, 47, 60, 61, 92, 96, 98, 107, 109, 113],
        )

        sites = replace(guide, guidance=replace(guide.guidance, mask_positions=[36, 7]))
        # Canonical order, so saved configs and library records compare.
        self.assertEqual(sites.guidance.mask_positions, [7, 36])
        self.assertEqual(sites, deserialize_config(serialize_config(sites)))

        for positions, message in (
            ([], "at least one site"),
            ([7, 7], "must not repeat"),
            ([0], "1..119"),
            ([120], "1..119"),
            ([True], "must be integers"),
        ):
            with self.subTest(positions=positions):
                with self.assertRaisesRegex(ValueError, message):
                    replace(
                        guide,
                        guidance=replace(guide.guidance, mask_positions=positions),
                    )

        # Either masking key alone loads; the two together are refused.
        text = re.sub(
            r"(?m)^mask_positions\s*=.*$",
            "",
            (ROOT / DEFAULT_CONFIGS["guidance"]).read_text(encoding="utf-8"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for suffix, expected in (
                ("mask_fraction = 0.1\n", 0.1),
                ("mask_positions = [7, 36]\n", None),
            ):
                path = Path(directory) / "one.toml"
                path.write_text(text + suffix, encoding="utf-8")
                cfg = load_config(path)
                if expected is None:
                    self.assertEqual(cfg.guidance.mask_positions, [7, 36])
                else:
                    self.assertIsNone(cfg.guidance.mask_positions)
                    self.assertEqual(cfg.guidance.mask_fraction, expected)
            path = Path(directory) / "both.toml"
            path.write_text(
                text + "mask_fraction = 0.1\nmask_positions = [7, 36]\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not both"):
                load_config(path)

    def test_modal_paths_and_external_checkpoint_are_explicit(self):
        cfg = load_config(ROOT / DEFAULT_CONFIGS["train-predictor"])
        remote = modal_config(cfg, "predictor-v1", "")
        payload = json.loads(serialize_config(remote))
        self.assertEqual(payload["csv_path"], "/data/creilov_combined.csv")
        self.assertEqual(payload["output_dir"], "/runs/predictor-v1")
        self.assertEqual(payload["device"], "cuda")
        guide = guided_config()
        self.assertEqual(
            modal_config(guide, "guide-v1", "").predictor_checkpoint,
            guide.predictor_checkpoint,
        )
        with self.assertRaises(ValueError):
            modal_config(guide, "guide-v1", "/runs/../outside/model.pt")
        checkpoint = "/runs/predictor-v1/predictor/ensemble_5/model.pt"
        remote = modal_config(guide, "guide-v1", checkpoint)
        self.assertEqual(
            json.loads(serialize_config(remote))["predictor_checkpoint"], checkpoint
        )
        self.assertEqual(
            json.loads(serialize_config(remote))["output_dir"], "/runs/guide-v1"
        )
        evaluation = load_config(ROOT / DEFAULT_CONFIGS["evaluate"])
        defaults = modal_config(evaluation, "eval-v1", "")
        self.assertEqual(defaults.library_path, evaluation.library_path)
        self.assertEqual(defaults.oracle_checkpoint, evaluation.oracle_checkpoint)
        remote = modal_config(
            evaluation,
            "eval-v1",
            checkpoint,
            "/runs/oracle/oracle/model.pt",
            "/runs/guide-v1/library.fasta",
        )
        self.assertEqual(remote, deserialize_config(serialize_config(remote)))
        self.assertEqual(remote.library_path.as_posix(), "/runs/guide-v1/library.fasta")


if __name__ == "__main__":
    unittest.main()
