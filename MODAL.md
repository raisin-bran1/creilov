# Running CreiLOV on Modal

Each successful data stage now automatically creates local CSV/PNG/Markdown
reports in `outputs/<run-id>/`. The `creilov-runs` Volume and `outputs/` share
one layout, `<stage>/<run-id>`, so a run's reports sit at the same path locally
as its full artifacts do on the Volume. A run ID may be nested, for example
`--run-id guidance/15-positions/tag-v1`, which groups related runs on both
sides. Evaluation has no stage folder of its own: it writes into the guidance
run holding the library it scored. Small JSON summaries are downloaded into
`.cache/report-inputs/<run-id>/`; checkpoints, split-index manifests and NPZ
prediction arrays stay on Modal. Figures render locally through `uv run` in
the project environment, so the isolated Modal CLI needs no plotting packages.

To refresh reports for an already completed run without training, sampling or
allocating a GPU worker:

```powershell
modal run modal_app.py --stage train-predictor --run-id train-predictor/predictor-v1 --reports-only
modal run modal_app.py --stage benchmark --run-id benchmark/predictor-benchmark-v1 --reports-only
modal run modal_app.py --stage evaluate --run-id guidance/full-unmask/control --reports-only
```

The report stage must match the saved run. Health only prints its status.
For a detached job, keep the local client open to receive automatic report
downloads; if it disconnects, run `--reports-only` after the remote job finishes.
A failed local render can also be retried this way without repeating compute.

| Stage | Local reports |
| --- | --- |
| `splits` | Split-size CSV, table PNG and Markdown |
| `benchmark` | Full metric CSV, performance table, Spearman table, split sizes |
| `train-predictor` | Ensemble validation metrics/tables, learning curves, split sizes; standard mask levels also get the existing ensemble comparison figures |
| `train-oracle` | Clean validation metrics/table, learning curves, split sizes |
| `guidance` | Library CSV |
| `evaluate` | Per-sequence scores, metric CSV/table/Markdown, both histograms and the evaluation dashboard, written into the guidance folder |

Every saved-stage report directory includes a `README.md` linking its files
and naming the source Modal run. The TOML used to launch the run is saved on
the run Volume and copied to `outputs/<run-id>/config.toml`, or to
`evaluate_config.toml` for an evaluate run, since that shares the guidance
folder and must not overwrite the generating config. CLI overrides
such as `--num-samples` and checkpoint paths are not in it; the resolved
values remain in the run's `config.json` on Modal. Runs launched before this
change have no saved TOML and skip the copy. Oracle scores remain in the evaluate stage.

The adapter in `modal_app.py` calls the existing workflow; training and guidance
still use ProteinGen. It forwards both local source trees, recreates Linux
dependencies from `uv.lock`, and uses Python 3.12 (required by ProteinGen).
No Windows `.venv`, Git directory, raw XLSX files, or local credentials are
uploaded. Only source directories and explicitly named build manifests are
included. The processed CSV is uploaded separately to a Volume.

The original adapter has completed a T4 benchmark. The split-stage adapter is
covered by local config, dispatch, and model tests; this refactor does not
automatically submit a cloud job. A health check verifies Linux dependencies,
ProteinGen imports and CUDA without training a model.

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

## Health check

```powershell
modal run modal_app.py --stage health
```

The health stage requires no TOML. You may use the installed `modal` command or
prefix it with `uvx --from modal==1.6.3`. Importing the launcher starts no paid
compute; invoking a remote stage does.

The launcher defaults to one T4, two CPU cores, 8 GiB RAM, a 30-minute
function timeout, no warm pool, and no configured application retries. Modal
infrastructure failures/preemptions may still have their own platform behavior.
It scales down after a short idle window. Don't launch multiple copies if you
want to keep only one GPU running. These are batch jobs, not deployed services.

Function timeout is not checkpoint resumption. The current training code saves
each role only after its ensemble finishes. A timeout during a role can lose
that role's progress; completed stages/roles remain on the output Volume.

## Separate stages

Each stage automatically chooses its corresponding config. Pass `--config` to
use a different file declaring the same `run.stage`. Benchmark references the
model/training sections of the two training files; these references are resolved
locally and sent as a frozen JSON payload. No referenced local paths need to
exist inside the Modal container.

```powershell
$env:CREILOV_MODAL_GPU = "T4" # also L4, A10, L40S
$env:CREILOV_MODAL_TIMEOUT = "10800" # example: allow up to three hours

# No oracle training is triggered by predictor training.
modal run --detach modal_app.py --stage train-predictor --run-id train-predictor/predictor-v1

# Optional independent training of a fresh S2 oracle.
modal run --detach modal_app.py --stage train-oracle

# Exactly eight single-model fits, with held-out test metrics.
modal run --detach modal_app.py --stage benchmark --run-id benchmark/v2

# Read a predictor from its run; write guidance artifacts to a nested run.
modal run --detach modal_app.py --stage guidance --num-samples 100 --method tag \
  --run-id guidance/15-positions/tag-v2
# For five members, select .../predictor/ensemble_5/model.pt instead.

# Download a single file; Volume and outputs/ paths are the same below the run.
modal volume get creilov-runs /guidance/15-positions/tag-v1/library.csv outputs/guidance/15-positions/tag-v1/library.csv
```

| Stage | Default config |
| --- | --- |
| `benchmark`, `splits` | `configs/benchmark.toml` |
| `train-predictor` | `configs/train_predictor.toml` |
| `train-oracle` | `configs/train_oracle.toml` |
| `guidance` | `configs/guidance.toml` |
| `evaluate` | `configs/evaluate.toml` |

The previous combined `train` stage has been replaced by `train-predictor` and
`train-oracle`. Predictor training retains nested sizes 1/5/20. Oracle training
retains ten members. Each training stage reports validation metrics only and
leaves the test data untouched. The optional oracle stage trains a fresh model;
it does not download the published ProteinGuide oracle.

`--detach` keeps a submitted batch running if the local client disconnects.
Check its dashboard logs; stop it there or with `modal app stop --yes <app-id>`.
Distinct run IDs preserve previous outputs. Defaults are `predictor`, `oracle`,
`benchmark`, and `guidance` for their respective stages. Using an existing run
ID overwrites matching artifacts.

The adapter selects CUDA, maps the input CSV basename to `/data`, and writes
outputs to `/runs/<run-id>`. All runtime paths in the checked-in stage TOMLs are
Modal paths: `/data/...` on `creilov-data` and `/runs/...` on `creilov-runs`.
Guidance and evaluation read their input paths directly from TOML; optional
`--predictor-checkpoint`, `--oracle-checkpoint`, and `--library-path` flags override
them. The default run ID is `run.output_dir` below `/runs`, so the TOML places
the run and `--run-id` overrides it. A run ID may contain `/` to nest related
runs, as in `--run-id guidance/15-positions/tag-v1`; each segment is letters,
digits, underscores and hyphens. A config whose `output_dir` is outside `/runs`
names no Volume run, so launching it needs an explicit `--run-id`. An evaluate
run takes its ID from the library it scores rather than from `output_dir`.
`root` and `[training_configs]` remain local config-resolution
settings, since the launcher resolves configuration before submission.
Guidance does not load an oracle. `--control` launches the unguided control
arm from the same TOML: it loads no predictor checkpoint, so
`--predictor-checkpoint` is rejected alongside it and its saved `config.json`
records `control = true` with no predictor checkpoint. Pair it with `--run-id`
to keep the control library separate from a guided one.
`--num-samples` and `--method tag|deg`
override the generation settings in its TOML. `guidance.batch_size` controls
batching. The stage writes `library.fasta`, `library.csv` and the resolved
config to its output run; a run that stops early simply has fewer rows than it
requested. Duplicates remain in the sampled library.

Evaluation adds `metrics.json`, `per_sequence.csv` and `evaluate_config.json`
to that same run folder, so a library and its scores never drift apart. Its
launch TOML is saved as `evaluate_config.toml`, leaving the generating run's
`config.toml` untouched. Figures are not written on the Volume at all: they
render locally, titled with the run ID.

For local execution, use a separate TOML with local paths and `device = "cpu"`.
The benchmark's S2-only data policy is unchanged.
Predictor training uses the deduplicated combined CSV. Generate it locally and
upload it separately; benchmark/oracle still use the existing S2 CSV:

```powershell
uv run python -m creilov.combined data
modal volume put creilov-data data/creilov_combined.csv /creilov_combined.csv
```

The guidance config uses the combined CSV to verify the new predictor's
training-data fingerprint. For an older S2-only predictor, select its original
S2 CSV in the guidance config instead.

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

For a small trial, copy the predictor config and reduce its ensemble size,
ensemble-size list, epochs, and row budget consistently. Measure wall time in the returned
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

Evaluate an existing library with saved checkpoints on `creilov-runs`:

```powershell
modal run modal_app.py --stage evaluate
```

Edit the paths in `configs/evaluate.toml` for your runs. All three inputs must already
exist on the volume; the evaluator does not train an oracle automatically.
The report and per-sequence scores are written to `creilov-runs:/evaluation`.
`configs/evaluate.toml` controls batch size and oracle histogram bins. Evaluation
does not need the original training CSV or load a generative model. The existing
Modal runner allocates the selected GPU; for these small MLPs, local CPU
evaluation with a local-path, CPU config avoids that cost entirely.
