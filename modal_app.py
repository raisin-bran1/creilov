"""Thin Modal adapter. Run with uvx --from modal==1.6.3 modal run modal_app.py.

Uses the existing uv.lock and sibling ProteinGen source, not a cloud rewrite.
Importing this file does not start a remote job. GPU jobs start at .remote().
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import modal

ROOT = Path(__file__).resolve().parent
PROTEINGEN = ROOT.parent / "proteingen"
GPU = os.environ.get("CREILOV_MODAL_GPU", "T4")
TIMEOUT = int(os.environ.get("CREILOV_MODAL_TIMEOUT", "3600"))
MEMORY_MB = int(os.environ.get("CREILOV_MODAL_MEMORY_MB", "8192"))
COMMIT_INTERVAL = 60  # seconds between run-volume commits while a stage runs
if GPU not in {"T4", "L4", "A10", "L40S"}:
    raise ValueError("CREILOV_MODAL_GPU must be T4, L4, A10 or L40S")
if not 60 <= TIMEOUT <= 86400:
    raise ValueError("CREILOV_MODAL_TIMEOUT must be between 60 and 86400 seconds")
if not 2048 <= MEMORY_MB <= 131072:
    raise ValueError("CREILOV_MODAL_MEMORY_MB must be between 2048 and 131072")

# Bake dependency manifests into image layers, then forward source at runtime.
# Excluding the two local packages from installation lets source edits reuse the
# expensive dependency layer. PYTHONPATH makes their local source importable.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "libgomp1")
    .uv_pip_install("uv==0.12.23")
)
for local, remote in (
    (ROOT, "/workspace/creilov"),
    (PROTEINGEN, "/workspace/proteingen"),
):
    for filename in ("pyproject.toml", "README.md"):
        image = image.add_local_file(
            local / filename, f"{remote}/{filename}", copy=True
        )
image = (
    image.add_local_file(ROOT / "uv.lock", "/workspace/creilov/uv.lock", copy=True)
    .run_commands(
        "cd /workspace/creilov && UV_PROJECT_ENVIRONMENT=/opt/venv "
        "uv sync --frozen --no-dev --no-install-project --no-install-package proteingen"
    )
    .env(
        {
            "PYTHONPATH": "/workspace/creilov/src:/workspace/proteingen/src",
            "HF_HOME": "/cache/huggingface",
            "TORCH_HOME": "/cache/torch",
            "MPLCONFIGDIR": "/cache/matplotlib",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
    )
    .add_local_dir(ROOT / "src", "/workspace/creilov/src")
    .add_local_dir(PROTEINGEN / "src", "/workspace/proteingen/src")
)

app = modal.App("creilov-proteinguide")
data_volume = modal.Volume.from_name("creilov-data", create_if_missing=True, version=2)
output_volume = modal.Volume.from_name(
    "creilov-runs", create_if_missing=True, version=2
)
cache_volume = modal.Volume.from_name(
    "creilov-cache", create_if_missing=True, version=2
)

DRIVER = """
from pathlib import Path
import sys
from creilov.logging_utils import quiet_unused_structure_mirrors
quiet_unused_structure_mirrors()
import torch
torch.set_num_threads(2)
assert torch.cuda.is_available(), 'CUDA is unavailable'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
from creilov.config import deserialize_config, validate_stage_config
from creilov.workflow import prepare, train_predictor, train_oracle, benchmark, generate_library, evaluate_library
stage, config_file, run_id = sys.argv[1:]
if stage == 'health':
    print('ProteinGen and CreiLOV imports succeeded; CUDA is available', flush=True)
else:
    cfg = deserialize_config(Path(config_file).read_text(encoding='utf-8'))
    validate_stage_config(stage, cfg)
    if stage != 'evaluate' and not cfg.csv_path.is_file():
        raise FileNotFoundError('Upload data/creilov.csv to creilov-data first')
    if stage == 'splits':
        prepare(cfg)
    elif stage == 'train-predictor':
        train_predictor(cfg)
    elif stage == 'train-oracle':
        train_oracle(cfg)
    elif stage == 'benchmark':
        benchmark(cfg)
    elif stage == 'guidance':
        generate_library(cfg)
    elif stage == 'evaluate':
        evaluate_library(cfg)
"""


@app.function(
    image=image,
    gpu=GPU,
    cpu=2,
    memory=MEMORY_MB,
    timeout=TIMEOUT,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    # No application retries. No stage checkpoints its progress, so a retry
    # restarts from zero rather than resuming, and a retry after a timeout just
    # spends a second GPU-hour hitting the same wall. A retry can also land on
    # the GPU before the cancelled container has released it, turning a timeout
    # into a confusing CUDA OOM. Raise CREILOV_MODAL_TIMEOUT instead.
    retries=0,
    volumes={"/data": data_volume, "/runs": output_volume, "/cache": cache_volume},
    # To use gated ESM3 weights, create this secret and uncomment the next line:
    # secrets=[modal.Secret.from_name("huggingface")],  # contains HF_TOKEN
)
def run_job(stage: str, config_text: str, run_id: str, toml_text: str = "") -> dict:
    started = time.monotonic()
    if toml_text:
        # Keep the launch TOML beside the run so reports can copy it locally.
        # Evaluation shares the guidance run folder, so it uses its own name.
        run_dir = Path("/runs") / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        name = "evaluate_config.toml" if stage == "evaluate" else "config.toml"
        (run_dir / name).write_text(toml_text, encoding="utf-8")
    try:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "resolved_config.json"
            config_path.write_text(config_text, encoding="utf-8")
            process = subprocess.Popen(
                [
                    "/opt/venv/bin/python",
                    "-u",
                    "-c",
                    DRIVER,
                    stage,
                    str(config_path),
                    run_id,
                ],
            )
            # Checkpoint written members periodically; a worker that dies
            # mid-stage never runs the finally block below.
            while process.poll() is None:
                try:
                    process.wait(timeout=COMMIT_INTERVAL)
                except subprocess.TimeoutExpired:
                    output_volume.commit()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, "driver")
    finally:
        # Preserve completed artifacts even if a later part of the stage fails.
        output_volume.commit()
        cache_volume.commit()
    return {
        "stage": stage,
        "run_id": run_id,
        "gpu": GPU,
        "elapsed_seconds": time.monotonic() - started,
        "artifacts": f"creilov-runs:/{run_id}",
    }


@app.local_entrypoint()
def main(
    stage: str = "health",
    config: str = "",
    run_id: str = "",
    predictor_checkpoint: str = "",
    oracle_checkpoint: str = "",
    library_path: str = "",
    num_samples: int = -1,
    method: str = "",
    control: bool = False,
    reports_only: bool = False,
):
    # These config helpers use only the standard library, so global modal/uvx
    # clients need no local ProteinGen installation to resolve role references.
    sys.path.insert(0, str(ROOT / "src"))
    from creilov.config import (
        DEFAULT_CONFIGS,
        default_run_id,
        load_config,
        modal_config,
        serialize_config,
        validate_stage_config,
        guidance_overrides,
        qualify_run_id,
    )
    from creilov.reports import check_run_id, sync_reports

    if stage not in {"health", *DEFAULT_CONFIGS}:
        raise ValueError(
            f"Unknown stage {stage!r}; choose health or {list(DEFAULT_CONFIGS)}"
        )
    if stage == "health":
        if reports_only:
            raise ValueError(
                "Health has no saved run; --reports-only needs another stage"
            )
        run_id = run_id or "health"
        if num_samples != -1 or method or control:
            raise ValueError("Generation options are only used by guidance")
        config_text = toml_text = ""
    else:
        toml_path = ROOT / (config or DEFAULT_CONFIGS[stage])
        toml_text = toml_path.read_text(encoding="utf-8")
        cfg = load_config(toml_path)
        validate_stage_config(stage, cfg)
        # The TOML's output_dir places the run on the Volume; --run-id moves it,
        # relative to the stage folder unless it already names one.
        run_id = (
            qualify_run_id(cfg, run_id, library_path)
            if run_id
            else default_run_id(cfg, library_path)
        )
        cfg = guidance_overrides(
            cfg, None if num_samples == -1 else num_samples, method or None, control
        )
        cfg = modal_config(
            cfg, run_id, predictor_checkpoint, oracle_checkpoint, library_path
        )
        config_text = serialize_config(cfg)
    check_run_id(run_id)
    if not reports_only:
        print(f"Starting {stage} on {GPU}, timeout={TIMEOUT}s, run={run_id}")
        print(run_job.remote(stage, config_text, run_id, toml_text))
    if stage != "health":
        print(f"Preparing local reports for {run_id}")
        print(sync_reports(output_volume, stage, run_id, ROOT))
