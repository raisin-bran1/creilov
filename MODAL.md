# Running CreiLOV on Modal

The adapter in `modal_app.py` calls the existing workflow; training and guidance
still use ProteinGen. It forwards both local source trees, recreates Linux
dependencies from `uv.lock`, and uses Python 3.12 (required by ProteinGen).
No Windows `.venv`, Git directory, raw XLSX files, or local credentials are
uploaded. Only source directories and explicitly named build manifests are
included. The processed CSV is uploaded separately to a Volume.

This setup has been checked locally for syntax and configuration, but has not
been built or run in a Modal account. The first health check verifies Linux
dependency installation, ProteinGen imports and CUDA before starting training.

## One-time account setup

From the project root in PowerShell:

```powershell
# Isolated Modal CLI; no change to the project's dependency environment.
uvx --from modal==1.6.3 modal setup

# Upload the processed S2 table once. Do not upload the Windows virtualenv.
uvx --from modal==1.6.3 modal volume create creilov-data --version=2
uvx --from modal==1.6.3 modal volume put creilov-data data/creilov.csv /creilov.csv
```

Modal currently requires a payment method for GPU use. In the Modal dashboard,
configure a Workspace usage budget that fits the included credit, and verify
the effective limit displayed there. Credits are an allowance, not an automatic
stop on pay-as-you-go usage. Account setup is described at
https://modal.com/docs/guide and budget semantics at
https://modal.com/docs/guide/budgets.

## First run: small pilot

```powershell
# Build the image and verify imports/CUDA, without training or loading ESM weights.
uvx --from modal==1.6.3 modal run modal_app.py --stage health

# Train two one-member MLPs on 2,000 rows each for at most three epochs.
uvx --from modal==1.6.3 modal run modal_app.py --stage train --run-id pilot

# Download checkpoints, timings/history, predictions and metrics.
uvx --from modal==1.6.3 modal volume get creilov-runs /pilot outputs/modal-pilot

# Optional: exercise actual foundation-model guidance with public ESM-C weights.
uvx --from modal==1.6.3 modal run modal_app.py --stage guidance --run-id pilot
```

`configs/modal_pilot.toml` is deliberately smaller than the reproduction setup.
The pilot uses ESM-C only for its optional guidance check; set `generator` to
`esm3-open` for the paper's generator after configuring Hugging Face access.
Neither the local launcher nor import starts a paid job until `.remote()` is
invoked by `modal run`. Running the health command does incur startup/compute.

The launcher defaults to one T4, two CPU cores, 8 GiB RAM, a 30-minute
function timeout, no warm pool, and no configured application retries. Modal
infrastructure failures/preemptions may still have their own platform behavior.
It scales down after a short idle window. Don't launch multiple copies if you
want to keep only one GPU running. These are batch jobs, not deployed services.

Function timeout is not checkpoint resumption. The current training code saves
each role only after its ensemble finishes. A timeout during a role can lose
that role's progress; completed stages/roles remain on the output Volume.

## Full runs

```powershell
# Change resources before importing the launcher.
$env:CREILOV_MODAL_GPU = "L4" # optional override of the T4 default; also A10, L40S
$env:CREILOV_MODAL_TIMEOUT = "3600"
# Consider 16 GiB host RAM for foundation-model loading (increases RAM cost):
# $env:CREILOV_MODAL_MEMORY_MB = "16384"

uvx --from modal==1.6.3 modal run --detach modal_app.py --stage train --config configs/workflow.toml --run-id mlp-v1
uvx --from modal==1.6.3 modal run --detach modal_app.py --stage benchmark --config configs/workflow.toml --run-id mlp-benchmark-v1
uvx --from modal==1.6.3 modal run modal_app.py --stage guidance --config configs/workflow.toml --run-id mlp-v1
```

`--detach` lets a submitted batch continue after the local client disconnects.
Check its logs in Modal's dashboard. Stop it there if necessary. Use distinct
run IDs for experiments; using an existing ID overwrites matching outputs.
Use the same run ID to load the predictor for a subsequent guidance check.
The adapter changes only device to CUDA, input path to `/data/creilov.csv`, and
output path to `/runs/<run-id>`; all modeling choices remain in normal TOML.
Edit the same TOML for oracle/predictor kinds, ensemble sizes, training budgets,
evaluation masking fractions and sample size, and TAG versus DEG. `train` fits
the two configured ensembles on `run.train_split` and reports validation metrics.
`benchmark` uses those same role-specific model/training configurations on all
four splits, overriding only ensemble size to one. This is eight single-model
fits, with no candidate-model sweep. Benchmark metrics use disjoint test data.

Run split generation locally (`uv run creilov splits`) to avoid renting a GPU
for CPU-only partitioning. The remote splits stage is available for debugging.

## Persistent data and model access

The launcher lazily creates `creilov-runs` (checkpoints/reports) and
`creilov-cache` (Hugging Face, Torch, matplotlib caches), in addition to the data
Volume. Cache and output writes are committed when the function exits normally
or through a caught stage failure. Hard termination need not run cleanup.
Model weights are cached after their first successful download.

The Modal driver filters only AtomWorks' two missing `CCD_MIRROR_PATH` and
`PDB_MIRROR_PATH` notices, which are irrelevant to this sequence-only workflow.
Other warnings and errors remain visible; neither variable is given a fake path.

Inspect stored files without starting a GPU job:

```powershell
modal volume list
modal volume ls creilov-data
modal volume ls creilov-runs /pilot
modal volume dashboard creilov-runs
```

For gated ESM3 access, accept the model's terms on Hugging Face, create a Modal
Secret named `huggingface` containing `HF_TOKEN`, and uncomment the `secrets=`
line on the function decorator. Create the Secret in Modal's dashboard rather
than putting a token in source or shell history. Public MLP training does not
need this Secret. See https://modal.com/docs/guide/secrets and
https://modal.com/docs/guide/volumes.

## Spending and efficiency

As checked on 2026-11-07, GPU-only prices are about $0.59/hour (T4), $0.80/hour
(L4), and $1.10/hour (A10). CPU and RAM are additional. At the launcher's
requested two cores and 8 GiB, L4 works out to approximately $0.96/hour before
any additional usage, image builds or other charges. Thus $30 is roughly 31
hours at that allocation, not a guarantee of 31 hours of useful training.
Always check current prices: https://modal.com/pricing.

Start with one ensemble member and the pilot. Measure wall time in the returned
job summary and inspect Modal's actual usage before multiplying seeds, ensemble
members. Small MLPs may be CPU/tokenization-bound; a more expensive GPU is not
automatically faster. `[evaluation].max_rows` can cap post-training validation
or benchmark test scoring; checkpoint selection still uses full validation.

For ESM probes, reduce training batch sizes initially (e.g. 4), use TAG first,
and wait until one run fits comfortably before increasing ensemble sizes. Frozen
weights still require activations for TAG input gradients. The current pipeline
already shares the backbone across ensemble heads during ensemble inference,
but does not cache clean training embeddings or pretokenize the entire dataset.
Those are possible future optimizations after profiling; masked embeddings
cannot simply reuse a clean embedding cache.

The build layer uses the existing frozen lock with the two local packages
excluded from installation; their runtime sources are on PYTHONPATH. This
preserves `../proteingen` without trying to install a different PyPI package.
The current ProteinGen package also pulls documentation and structure-model
dependencies even for MLP-only work. The first image build can therefore be
large. It is cached independently of source edits. If the Linux build fails,
fix the reported dependency conflict rather than silently switching model or
ProteinGen versions. See https://modal.com/docs/guide/custom-container.
