# CreiLOV

Reimplementing the CreiLOV workflow from ProteinGuide using the local ProteinGen
library. See [data/README.md](data/README.md) for sources, preprocessing, and
ProteinDataset usage.

For GPU batch jobs with persistent outputs and a small pilot configuration, see
[MODAL.md](MODAL.md). The cloud adapter runs this same workflow and ProteinGen.

```powershell
uv run python -m creilov.data data
uv run python -m unittest discover -s tests -v
```

## Configurable ProteinGuide workflow

Edit [configs/workflow.toml](configs/workflow.toml), then run from the project root:

```powershell
# Generate all four split manifests and inspect their sizes first.
uv run creilov splits --config configs/workflow.toml

# Benchmark configured oracle/predictor on every split, one member per role.
uv run creilov benchmark --config configs/workflow.toml

# Train the final clean oracle and noisy guidance predictor on run.train_split.
uv run creilov train --config configs/workflow.toml

# Load the trained noisy predictor and check a real ESM TAG/DEG step.
uv run creilov guidance --config configs/workflow.toml
```

`uv run creilov all` runs training and guidance, without automatically launching
a benchmark. `train` fits only the two configured ensembles on `run.train_split`.
`benchmark` makes eight single-member fits: four splits x two roles. It copies
each role's architecture and training settings, overriding only ensemble size
to one; there is no candidate architecture sweep. Commands overwrite reports in the
configured output directory. Use a new output directory to preserve a run.
Paths resolve relative to `run.root`, itself relative to the config file.

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

The local default is CPU, a ten-member MLP oracle and five-member noisy predictor.
Modal always overrides `run.device` to `cuda`; the resolved value is saved in
`config.json`. The CPU setting does not disable the rented GPU.
The latter uses 2,000 training rows. Nothing automatically downloads ESM weights
for MLP training. Enabling a probe or running the guidance command loads the
named pretrained checkpoint; ESM3 may require Hugging Face access. Use a GPU and
smaller training batch sizes (for example 4) for foundation-model probes. No
clean embedding cache is reused for noisy training, because the input masks
change. CPU foundation-model inference is supported but potentially slow.

### Split definitions

Every split partitions all unique S2 sequences into disjoint train/validation/test
sets. A fixed seed determines the random split and any training-budget subset.
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
validation/test partition. `train` uses `run.train_split` (random by default)
for both roles, retaining each configured ensemble size. It reports validation
metrics and leaves the test partition untouched. `benchmark` repeats training
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
`agreement.json` compares oracle/predictor scores on their common clean validation
sample, reusing already computed predictions. Validation reports are diagnostics
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

config = load_config(Path("configs/workflow.toml").resolve())
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

The CLI guidance check starts with configurable random masking over all 119
residues, forces the requested position masked, and writes base/guided amino
acid probabilities to `guidance_check.json`. It does not generate a completed
library. For DEG, downstream sampling must select a position via
`guided.at_position([token_position, ...])` (ProteinGen's any-order sampler does
this). Biological position 7 is token index 7 after adding BOS.

S2-only data, omission of family classification, ESM tokenization of the MLP,
validation-based checkpoint selection, and the residual variance floor are
explicit differences from an exact reproduction of every paper setting.
This implementation stops at a usable guidance model and a one-step check;
wet-lab validation, structure evaluation and library generation are later steps.
