"""Thin Modal adapter. Run with uvx --from modal==1.6.3 modal run modal_app.py.

Uses the existing uv.lock and sibling ProteinGen source, not a cloud rewrite.
Importing this file does not start a remote job. GPU jobs start at .remote().
"""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

import modal

ROOT = Path(__file__).resolve().parent
PROTEINGEN = ROOT.parent / "proteingen"
GPU = os.environ.get("CREILOV_MODAL_GPU", "T4")
TIMEOUT = int(os.environ.get("CREILOV_MODAL_TIMEOUT", "1800"))
MEMORY_MB = int(os.environ.get("CREILOV_MODAL_MEMORY_MB", "8192"))
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
from dataclasses import replace
from pathlib import Path
import sys
from creilov.logging_utils import quiet_unused_structure_mirrors
quiet_unused_structure_mirrors()
import torch
torch.set_num_threads(2)
assert torch.cuda.is_available(), 'CUDA is unavailable'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
from creilov.config import load_config
from creilov.workflow import prepare, train, benchmark, guidance_check
stage, config_file, run_id = sys.argv[1:]
if stage == 'health':
    print('ProteinGen and CreiLOV imports succeeded; CUDA is available', flush=True)
else:
    cfg = replace(load_config(Path(config_file)),
                  csv_path=Path('/data/creilov.csv'),
                  output_dir=Path('/runs') / run_id, device='cuda')
    if not cfg.csv_path.is_file():
        raise FileNotFoundError('Upload data/creilov.csv to creilov-data first')
    if stage == 'splits':
        prepare(cfg)
    elif stage == 'train':
        train(cfg)
    elif stage == 'benchmark':
        benchmark(cfg)
    elif stage == 'guidance':
        guidance_check(cfg)
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
    retries=0,
    volumes={"/data": data_volume, "/runs": output_volume, "/cache": cache_volume},
    # To use gated ESM3 weights, create this secret and uncomment the next line:
    # secrets=[modal.Secret.from_name("huggingface")],  # contains HF_TOKEN
)
def run_job(stage: str, config_text: str, run_id: str) -> dict:
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "workflow.toml"
            config_path.write_text(config_text, encoding="utf-8")
            subprocess.run(
                [
                    "/opt/venv/bin/python",
                    "-u",
                    "-c",
                    DRIVER,
                    stage,
                    str(config_path),
                    run_id,
                ],
                check=True,
            )
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
    config: str = "configs/modal_pilot.toml",
    run_id: str = "pilot",
):
    if stage not in {"health", "splits", "train", "benchmark", "guidance"}:
        raise ValueError("stage must be health, splits, train, benchmark, or guidance")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError(
            "run-id must contain only letters, digits, underscores and hyphens"
        )
    config_text = (ROOT / config).read_text(encoding="utf-8")
    print(f"Starting {stage} on {GPU}, timeout={TIMEOUT}s, run={run_id}")
    print(run_job.remote(stage, config_text, run_id))
