# CreiLOV

Reimplementing the CreiLOV workflow from ProteinGuide using the local ProteinGen
library. See [data/README.md](data/README.md) for sources, preprocessing, and
ProteinDataset usage.

For GPU batch jobs with persistent outputs, see
[MODAL.md](MODAL.md). The cloud adapter runs this same workflow and ProteinGen.

```powershell
uv run python -m creilov.data data
uv run python -m unittest discover -s tests -v
```

## Configurable ProteinGuide workflow

Each stage has its own config and default output directory:

| Stage | Config | Modal output (`creilov-runs`) |
| --- | --- | --- |
| `benchmark` | [configs/benchmark.toml](configs/benchmark.toml) | `/runs/benchmark/default` |
| `train-predictor` | [configs/train_predictor.toml](configs/train_predictor.toml) | `/runs/train-predictor/default` |
| `train-oracle` | [configs/train_oracle.toml](configs/train_oracle.toml) | `/runs/train-oracle/default` |
| `guidance` | [configs/guidance.toml](configs/guidance.toml) | `/runs/guidance/default` |
| `evaluate` | [configs/evaluate.toml](configs/evaluate.toml) | the scored library's own run |

The `creilov-runs` volume is laid out exactly like local `outputs/`:
`/runs/<stage>/<run-id>`, where a run ID may itself be nested to group related
runs (`/runs/guidance/15-positions/tag-v1`). A run's `output_dir` in its TOML
places it, and `--run-id` overrides that; each volume folder holds everything
its `outputs/` counterpart does, plus the checkpoints and arrays that stay on
Modal.

Run from the project root; each command automatically selects the config above:

```powershell
# Generate all four split manifests and inspect their sizes first.
modal run modal_app.py --stage splits

# Benchmark configured oracle/predictor on every split, one member per role.
modal run modal_app.py --stage benchmark

# Train only the noisy predictor, including its nested 1/5/20-member ensembles.
modal run modal_app.py --stage train-predictor

# Optional: train a fresh clean oracle independently.
# (ProteinGuide's published oracle is also available; see the note below.)
modal run modal_app.py --stage train-oracle

# Generate a library using the trained noisy predictor.
modal run modal_app.py --stage guidance --num-samples 100 --method tag
```

`--config <path>` overrides a stage's default file. Configs declare `run.stage`;
using a config for the wrong stage raises an error. The former `workflow.toml`,
combined `train`, and `all` commands have been replaced by the separate stages.
Guidance uses `run.predictor_checkpoint`, so its output folder can differ from
the predictor's training folder. It does not require a trained oracle.

### Evaluation oracle

`evaluate` defaults to ProteinGuide's published CreiLOV oracle, ported into this
project's checkpoint format at `creilov-runs:oracle-proteinguide/oracle/model.pt`.
It is a 10-member MLP ensemble fit to all 167,530 assay rows, in the same log10
fluorescence units as our labels, so `guidance.threshold` transfers unchanged and
the paper's numbers are comparable. Point `run.oracle_checkpoint` at
`/runs/oracle/oracle/model.pt` instead to judge with our own `train-oracle`
checkpoint; agreement between the two is a useful diagnostic.

Regenerate the port with [scripts/port_proteinguide_oracle.py](scripts/port_proteinguide_oracle.py).
The mapping, verification gates and known limitations — notably that the ported
oracle is valid for clean sequences only and its variance channel is not
calibrated — are documented in
[docs/proteinguide_oracle_port.md](docs/proteinguide_oracle_port.md).

`benchmark` makes eight single-member fits: four splits x two roles. Its
`[training_configs]` references the two training TOMLs, importing each role's
model and optimizer/training settings while overriding ensemble size to one.
It uses its own S2 data, split, evaluation and calibration settings; changing
training output paths or nested ensemble sizes does not launch additional
benchmark fits. The references are resolved before submission to Modal.
There is no candidate architecture sweep. Commands overwrite reports in the
configured output directory. Use a new output directory to preserve a run.
Paths resolve relative to `run.root`, itself relative to the config file.

Predictor training now uses `data/creilov_combined.csv`: all S2 rows plus the
WT-aligned S1 rows whose sequences are absent from S2. Generate it with
`uv run python -m creilov.combined data` after exporting aligned S1. This yields
167,530 unique sequences (165,428 S2 and 2,102 additional S1). The random split
is formed before uniformly selecting 2,000 training rows without replacement;
there is no library quota. All predictor ensemble sizes share these rows.
Benchmark and fresh-oracle configs continue to use S2-only `data/creilov.csv`.

### Models and training

- `oracle.kind` and `predictor.kind` independently select `one_hot_mlp`,
  `linear_probe`, or `additive`. Each has an independent `ensemble_size`.
- `hidden_layers` counts hidden linear layers. One hidden layer matches the
  two-total-linear-layer architecture used in the paper's CreiLOV MLP.
- `linear_probe` uses frozen contextual ESM-C or ESM3 representations and a
  trained linear head. Pooling averages residue positions including masks,
  excluding BOS/EOS/PAD/UNK. Gradients still pass through the frozen backbone
  into one-hot input for TAG. Probe ensemble members share the frozen backbone;
  ensemble inference extracts contextual features once, then applies all heads.
- `additive` is a positional one-hot linear regressor, a useful interaction-free
  baseline. There is no family classifier or family-labeled negative data.
- `oracle_training` and `predictor_training` control epochs, batch size,
  learning rate, weight decay, patience, training-row budget, and gradient clip.
  A budget of `0` uses all training rows. Early stopping selects each member's
  best **clean validation** MSE. Test labels never select checkpoints.
- The oracle sees clean sequences. The predictor uses ProteinGen's
  `uniform_mask_noise` and `uniform_time`: a separate random masking level per
  example per epoch. Original fluorescence labels remain unchanged. BOS/EOS
  and other special tokens are protected.
- Labels are standardized with statistics of the selected training rows only.
  Saved predictions and guidance thresholds are in original log10 fluorescence
  units. Members have independent initializations and masking/shuffle seeds;
  they use the same selected training rows, not bootstrap resamples.

Checked-in stage configs use Modal volume paths and CUDA, a ten-member MLP
oracle and twenty-member noisy predictor. `/data` is `creilov-data`; `/runs` is
`creilov-runs`. A local run needs a separate config with local paths and device.
Local `outputs/` contains report CSV, PNG and Markdown files only; full runs
and checkpoints stay on Modal. See [outputs/README.md](outputs/README.md).
Successful Modal stages automatically refresh local reports in
`outputs/<run-id>/`, mirroring the volume; a run ID may be nested, such as
`--run-id guidance/15-positions/tag-v1`. Evaluation has no run of its own: it
reports into the guidance folder holding the library it scored. Use
`modal run modal_app.py --stage <stage> --run-id <id>
--reports-only` to refresh a completed run without launching compute. Small
report inputs are cached under `.cache/report-inputs/`; plotting runs locally
in the project's uv environment. See [MODAL.md](MODAL.md) for report contents
and disconnected-client behavior.
The latter uses 2,000 training rows. Nothing automatically downloads ESM weights
for MLP training. Enabling a probe or running the guidance command loads the
named pretrained checkpoint; ESM3 may require Hugging Face access. Use a GPU and
smaller training batch sizes (for example 4) for foundation-model probes. No
clean embedding cache is reused for noisy training, because the input masks
change. CPU foundation-model inference is supported but potentially slow.

`run.predictor_ensemble_sizes = [1, 5, 20]` saves nested predictor ensembles
after training the twenty members once. Sizes must be increasing and end at
`predictor.ensemble_size`; an empty list disables the comparison. All sizes use
the same training rows and the first N trained members, with a separate
clean-validation residual variance estimate for each size. These are correlated
ensemble-size comparisons, not independent repeated runs. Benchmark still uses
one member per role and split.

The largest predictor stays at `predictor/model.pt` for guidance. Smaller
checkpoints and validation reports go in `predictor/ensemble_1/` and
`predictor/ensemble_5/`. `predictor_ensemble_summary.json` indexes all three
sizes, their checkpoint paths, member indices and validation metrics. The
`train-predictor` command leaves the test set untouched.

### Split definitions

Every split partitions unique sequences into disjoint train/validation/test
sets: S2 only for benchmark/oracle, combined S1+S2 for predictor training.
A fixed seed determines the random split and any training-budget subset.
Oracle and predictor share partitions, with independently configured row budgets.

| Split | Training | Validation | Test |
| --- | --- | --- | --- |
| Random | First configured fraction after shuffling | Next fraction | Remainder |
| Distance | Mutation count <= training cutoff | Counts above training cutoff through validation cutoff | Larger counts |
| Activity | Fitness <= training quantile | Fitness up through validation quantile | Higher tail |
| Position | WT at all held-out sites | Mutation at a validation site, none at test sites | Any mutation at a test site |

Position splits allow other observed mutations in the background. A test-site
mutation always takes precedence if a variant also mutates a validation site.
Held-out sites use 1-based biological numbering and must vary in S2. These tests
measure transfer within S2's 15 variable positions, not empirical validity at the
104 other positions. Activity and distance are separate stress tests; they may
also change the distribution of other properties. Empty partitions are errors.

With `max_train = 0`, the oracle uses the full **training partition**, never the
validation/test partition. Each training stage uses its own `run.train_split`
(random by default), retaining that role's configured ensemble size. It reports
validation metrics and leaves the test partition untouched. `benchmark` repeats training
for each split with ensemble size one and reports on each disjoint test set;
validation still controls checkpoint selection and residual-variance calibration.
No model is refitted on evaluation data.

### Reports and checkpoints

Outputs include resolved `config.json`, CSV SHA256-bound `splits.json`,
`split_summary.json`, and per-role training histories/checkpoints. Normal training
reports go to `<role>/validation/` and `validation_summary.json`. Benchmark
results go to `benchmark/<split>/<role>/test/`; `benchmark/summary.json`
collects them. Every benchmark checkpoint has one member, including for probes
and additive models. Its metadata records the actual model configuration and
evaluation indices. Each evaluation mask fraction gets a compressed prediction
table (`.npz`) containing sequences, labels, means and log variances.

Metrics include RMSE, MAE, Spearman, Gaussian NLL, threshold-event Brier score,
95% interval coverage, and top-k precision/recall/enrichment. Reports also break
down metrics by mutation count. Undefined correlations or recall/enrichment when
there are no positive labels are JSON `null`, not zero. Masked metrics use a
fixed evaluation seed and measure a single repeat of independent masking.
Each independent training run writes a `validation_summary.json` for its own
role. Cross-role agreement is no longer computed automatically by training.
Validation reports are diagnostics
on data used for checkpoint selection/calibration, not independent final-test
estimates. Agreement is not proof of real-world extrapolation.

`[evaluation]` configures `mask_fractions`, `top_k`, and `max_rows`. The oracle is
scored only on clean inputs; the noisy predictor is scored on clean and specified
masked inputs. Mask fractions must start with `0.0`. `max_rows = 0` uses all
evaluation rows. A positive limit selects a shared fixed-seed random sample,
preserving the full validation set for early stopping/calibration. Training
records scored rows in `validation_indices.json`; benchmark stores test indices
in per-role metadata. The old `[benchmark.models]` list has been removed.

Clean-validation residual MSE is used as a shared variance floor, added to
population variance across ensemble means. `guidance.min_sigma` sets a lower
bound. This supports single-member models and avoids degenerate tail
probabilities; it is an explicit adaptation of the paper's ensemble variance
heuristic. It is not a calibrated guarantee, especially on masked/OOD inputs.
Use Brier, coverage and NLL reports to assess it. Foundation-model checkpoints
store trained heads and model IDs, not copies of frozen pretrained weights.

### Guidance API

```python
from pathlib import Path
from creilov.config import load_config
from creilov.workflow import build_guidance

config = load_config(Path("configs/guidance.toml").resolve())
guided = build_guidance(config)  # native ProteinGen TAG or DEG
```

`guidance.method` selects TAG or DEG. Both use a Gaussian survival probability
`P(fitness > threshold)` computed from the noisy ensemble. Predictor parameters
are frozen for inference, but one-hot input gradients remain available.
TAG differentiates through the ensemble and, for probes, the full foundation
model. DEG evaluates every vocabulary choice at the selected position and
relies on the generator's formatter to rule out invalid tokens. The workflow
uses matching ESM vocabularies (including BOS/EOS), so no ad hoc token conversion
or undocumented cross-tokenizer DEG path is involved.

`generator_temperature` changes the base distribution; `predictor_temperature`
uses ProteinGen's native guidance temperature semantics. For TAG it scales the
Taylor correction; DEG applies temperature to the predictor's binary logits.
Set both predictor temperatures to 1 for the ordinary probability composition.

The guidance stage now generates a sequence library using native ProteinGen
`sample(..., n_parallel=1, in_order=None)`: one randomly ordered position is
revealed per iteration. Both TAG and DEG use this same sampler. It generates
119-residue sequences in batches; BOS/EOS remain protected.

Masking is set one of two ways, and `[guidance]` accepts exactly one of them:

- `mask_positions` names the designable sites in 1-based biological numbering.
  Every draw masks that same set, and unlisted positions hold their wild-type
  residue, so a library differs from CreiLOV by at most `len(mask_positions)`
  mutations. TOML has no null, so omitting the key is how "mask everything" is
  written. The shipped config uses the 15 sites the si_002 (S2) library varies:
  `[3, 4, 5, 7, 29, 34, 47, 60, 61, 92, 96, 98, 107, 109, 113]`.
- `mask_fraction` draws an independent random site set per draw;
  `mask_fraction = 1.0` masks every residue.

Prefer `mask_positions` for libraries meant to be comparable to the assay. Both
the predictor and the oracle were trained on variants within 11 mutations of
wild type, and full masking produces de novo proteins roughly 110 mutations
away, where neither model carries signal and guidance has nothing to act on.

```powershell
modal run modal_app.py --stage guidance --num-samples 100 --method tag
modal run modal_app.py --stage guidance --num-samples 20 --method deg
```

`--num-samples` and `--method` override `[guidance].num_samples` and `method`.
`batch_size` controls generation batch size in TOML. These options are rejected
for non-guidance stages. Defaults are 32 samples, batch size 4, and TAG. The
config points to `/runs/train-predictor/predictor-v1/predictor/model.pt` on the
Modal volume.

### Unguided control arm

`--control`, or `[guidance].control = true`, replaces the predictor with an
uninformative one: equal binary logits for every sequence, so its input
gradient is exactly zero. TAG then adds a zero delta and DEG adds the same
constant to every candidate token, leaving the generator's distribution
unchanged. The arm still runs through the same sampler, so at a shared seed it
draws the same reveal order as a guided run and isolates the predictor's
contribution.

```powershell
modal run modal_app.py --stage guidance --control --run-id guidance-control
```

Both arms share [configs/guidance.toml](configs/guidance.toml); pass a run ID
so the control does not overwrite a guided library. A control run drops
`run.predictor_checkpoint` from its saved config however it was launched, and
`control` is rejected outside the guidance stage, so a run's own config says
which arm it is.

Outputs in `run.output_dir`:

- `library.fasta`: all sampled sequences, each with a stable unique record ID.
- `library.csv`: ID, sequence, mutation count relative to WT, and
  `log_prob_given_order` from the native sampling trajectory. This is the
  sampler log probability conditional on its reveal order, not a fluorescence
  measurement or a marginal sequence likelihood.
- `config.json`: resolved settings including CLI overrides, which record the
  method, masking, seed and the predictor checkpoint (or the control arm).

No separate summary file is written: the counts it held are the rows of
`library.csv`, and the rest is in `config.json`. A run that stopped early
leaves fewer rows than `guidance.num_samples`, which is how reporting detects
an incomplete library and refuses to chart it.

Locally, the guidance stage exports only `library.csv` and its launch TOML to
`outputs/guidance/<run-id>/`. Scores, mutation statistics and figures for that
library are written into the same folder later by the evaluate stage, so each
library and its measurements stay together.

The requested count is the number of draws, not a guarantee of unique proteins.
Duplicates are retained and the unique count is reported. No oracle evaluation
is performed. The Python `guidance_check(cfg)` function remains available for
the old single-position probability diagnostic; the CLI runs library generation.

Our data/split policy, omission of family classification, ESM tokenization of the MLP,
validation-based checkpoint selection, and the residual variance floor are
explicit differences from an exact reproduction of every paper setting.
The generation loop uses random-order ancestral sampling. Wet-lab validation
and structure evaluation are separate later steps.

Evaluate a generated library with `modal run modal_app.py --stage evaluate`. Edit
[`configs/evaluate.toml`](configs/evaluate.toml) to select a FASTA or CSV library,
the clean oracle checkpoint produced by `train-oracle`, the noisy predictor
checkpoint used for guidance, and a separate output directory. The default
uses CUDA and the existing 20-member predictor; it does not train or generate.
The oracle must already exist at the configured path. This loader accepts this
project's checkpoints, not the published ProteinGuide weight-file format.

Evaluation writes `metrics.json`, `per_sequence.csv` and `config.json` to its
own Modal run directory. Local reporting then renders `per_sequence.csv`, an
`evaluation_metrics` table, two standalone histograms
(`oracle_scores_histogram.png`, `mutations_histogram.png`) and a
screenshot-ready `evaluation_summary.png` with both histograms, a correlation
scatter plot and the numeric table. These land in
`outputs/guidance/<guidance-run-id>/`, next to the `library.csv` they scored,
and every figure is titled with that guidance run ID so a loose PNG is still
traceable to its library. The evaluate launch TOML is copied alongside as
`evaluate_config.toml`, leaving the guidance `config.toml` intact. Pearson and Spearman correlations compare
predictor and oracle ensemble means on the complete, unmasked generated library.
They are null for fewer than two samples or constant scores. Oracle scores are
in log10 fluorescence units; variances are population variances across library
draws, rather than uncertainty within an ensemble. Mutation counts compare all
119 residues against wild type.

Following [ProteinGuide's metrics code](https://github.com/junhaobearxiong/proteinguide/blob/main/pdg/fitness/metrics.py),
diversity is the mean Hamming distance between unordered pairs of **unique**
sequences, in differing residues (not normalized by length). It is zero for a
single unique sequence. The implementation computes this exact mean from
per-position counts in linear time without constructing a pairwise matrix.
Other statistics and histograms retain duplicate library draws. Scoring reuses
ProteinGen's `ProteinDataset`, collators and predictive model interface; its
documented general diversity helper is currently marked “coming soon.”
