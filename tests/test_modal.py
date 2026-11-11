"""Exercise the Modal driver offline, without the Modal SDK or GPU jobs."""

import ast
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from creilov.config import DEFAULT_CONFIGS, load_config, serialize_config

ROOT = Path(__file__).parents[1]


class ModalDriverTests(unittest.TestCase):
    def test_remote_dispatch_runs_exactly_the_requested_stage(self):
        module = ast.parse((ROOT / "modal_app.py").read_text())
        driver = next(
            ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "DRIVER" for t in node.targets)
        )
        functions = {
            "splits": "prepare",
            "benchmark": "benchmark",
            "train-predictor": "train_predictor",
            "train-oracle": "train_oracle",
            "guidance": "generate_library",
            "evaluate": "evaluate_library",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "creilov.csv"
            csv_path.write_text("fixture")
            for stage, selected in functions.items():
                with self.subTest(stage=stage):
                    cfg = replace(
                        load_config(ROOT / DEFAULT_CONFIGS[stage]),
                        csv_path=csv_path,
                        output_dir=root / stage,
                        device="cuda",
                    )
                    config_path = root / "resolved.json"
                    config_path.write_text(serialize_config(cfg))
                    with ExitStack() as stack:
                        stack.enter_context(
                            patch(
                                "sys.argv",
                                ["driver", stage, str(config_path), "run-id"],
                            )
                        )
                        stack.enter_context(
                            patch("torch.cuda.is_available", return_value=True)
                        )
                        stack.enter_context(
                            patch("torch.cuda.get_device_name", return_value="test GPU")
                        )
                        stack.enter_context(patch("torch.set_num_threads"))
                        calls = {
                            name: stack.enter_context(patch(f"creilov.workflow.{name}"))
                            for name in functions.values()
                        }
                        exec(driver, {"__name__": "__main__"})
                        for name, mocked in calls.items():
                            if name == selected:
                                mocked.assert_called_once_with(cfg)
                            else:
                                mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
