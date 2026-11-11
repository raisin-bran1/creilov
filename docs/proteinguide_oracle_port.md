# Porting the ProteinGuide CreiLOV oracle

The `evaluate` stage scores a generated library with a *clean oracle*. This
document records how ProteinGuide's published CreiLOV oracle was brought into
this project's checkpoint format, what was verified, and which choices were
judgement calls.

Reproduce with:

```powershell
uv run python scripts/port_proteinguide_oracle.py `
  --checkpoint-dir <extracted>/checkpoints/creilov/oracle `
  --their-csv      <extracted>/data/creilov_processed.csv `
  --our-csv        data/creilov_combined.csv `
  --output         checkpoints/oracle-proteinguide/oracle/model.pt
```

## Why port it at all

An oracle we train ourselves sees essentially the same assay rows our guidance
predictor saw, so the judge and the student share a teacher. ProteinGuide's
oracle is an independent judge, trained on the full assay table, and it is the
model the paper's reported numbers were produced with — so using it makes our
results comparable to theirs rather than merely internally consistent.

It is intended as an *additional* oracle, not a replacement. Agreement between
it and our own `train-oracle` checkpoint is itself a useful diagnostic, and
`evaluate` takes `oracle_checkpoint` as a path, so running both is two
invocations.

## Provenance

| Item | Value |
| --- | --- |
| Source | ProteinGuide, `pdg/fitness` ([repo](https://github.com/junhaobearxiong/proteinguide)) |
| Weights | `fitness.tar.gz`, [Zenodo record 18423115](https://zenodo.org/records/18423115) |
| Archive SHA-256 | `b2ee9eb9d7ac4aa0bad6d5e6e1c99fa38f986f64d117aa9b8dca222c8087ace3` |
| Archive size | 426,911,980 bytes |
| Members | `checkpoints/creilov/oracle/model{0..9}.pth` |
| Their architecture | `fc1 (400, 2380)` → ReLU → `fc2 (1, 400)`, mean-only (MSE) |
| Their alphabet | `ACDEFGHIKLMNPQRSTVWY` (from `pdg/fitness/constants.py`) |
| Training data | `creilov_processed.csv`, 167,530 rows |
| Ported to | `checkpoints/oracle-proteinguide/oracle/model.pt` (63,934,144 bytes) |
| Modal path | `creilov-runs:train-oracle/oracle-proteinguide/oracle/model.pt` → `/runs/...` in-container |

Only `checkpoints/creilov/oracle/` and `data/creilov_processed.csv` were
extracted from the archive. The checkpoint is gitignored (`/checkpoints/`) and
lives on the Modal Volume; `model.port.json` beside it records the verification
report and the per-member source hashes.

## The port is a weight remap, not a re-implementation

Their member and our `FluorescenceMLP` are the **same topology**:

```
theirs: Linear(119*20 = 2380 -> 400) -> ReLU -> Dropout(0.1) -> Linear(400 -> 1)
ours:   Linear(121*33 = 3993 -> 400) -> ReLU -> Dropout(0.1) -> Linear(400 -> 1)
```

Our project defaults (`hidden_dim=400`, `hidden_layers=1`, `dropout=0.1`) already
match theirs. Only the one-hot *layout* differs: they encode 119 residues over 20
canonical letters; we encode 121 tokens (BOS + 119 + EOS) over the 33-token ESM
vocabulary. Both flatten position-major, so the port reshapes their
`[400, 119, 20]` input weights and scatters them into a zeroed `[400, 121, 33]`
at token offset `+1` and each residue's ESM vocab column. Layers after the first
are copied verbatim.

## Design choices

1. **Reuse `kind="one_hot_mlp"`; no new model class.** The topology is already
   identical, so a new class would be duplicate code and a second thing to keep
   in sync with `load_checkpoint`.
2. **Zero every column with no counterpart** — BOS, EOS, and the non-canonical
   ESM tokens. This is *exact*, not merely harmless: BOS/EOS are always-on
   constants, so a zero weight column contributes nothing and no bias correction
   is needed. Verified by test.
3. **Map their `X` onto our `<mask>` column.** `constants.py` uses `X` as the
   mask slot. The real checkpoints turned out to be 20-wide with no `X` column,
   so this path is inert here — it is retained so a 21-wide checkpoint would port
   correctly rather than silently mis-map.
4. **Disable label standardization** (`label_mean=0`, `label_scale=1`). Our
   ensemble standardizes targets during training and undoes it in
   `member_predictions`; their model predicts raw units. Leaving our training-set
   moments in place would skew every score.
5. **Do not calibrate `residual_variance`;** leave it at the `min_sigma` floor.
   Their members are mean-only, so there is no honest variance to port. This is
   sufficient because `evaluate` reads only channel 0, and the checkpoint is
   stamped `role="oracle"`, which makes `build_guidance` refuse it — so the
   uncalibrated variance cannot leak into guidance. Covered by a test.
6. **Keep `dropout=0.1`** even though `eval()` disables it: ProteinGen's
   `OneHotMLP` only emits a `Dropout` module when `dropout > 0`, and that shifts
   the `nn.Sequential` indices the state dict keys depend on.
7. **Detect their layers by shape, not by name.** Their attribute names are not
   a stable interface; the shapes are pinned by the problem (fan-in divisible by
   119, scalar output). The script refuses deeper nets and two-output
   heteroscedastic heads rather than guessing.

## Verification

A wrong alphabet permutation silently yields a confident, meaningless model, so
correctness was checked from independent directions. The script enforces these
and exits non-zero on failure.

| Gate | Result | Threshold |
| --- | --- | --- |
| Remap exactness vs. their forward pass replayed from the raw state dict | max abs deviation **4.8e-07** | < 1e-4 |
| Anchor: Spearman vs. their `fitness` column (4,000 rows) | **+0.8716** | ≥ 0.80 |
| Wild-type point prediction | **3.99405** vs. measured **4.02323** (error **0.029**) | ≤ 0.10 |
| Units: Spearman vs. *our* `fitness` column | +0.8630 | reported only |

**Why those thresholds.** Permutation controls establish the floors. Random
alphabet permutations score −0.53 to +0.02 Spearman (versus +0.872 for the true
order) and mispredict WT by 0.7–2.1. The WT check is the sharpest discriminator
because all 119 positions contribute: even a single A↔L transposition errs by
0.28, ten times the true order's 0.029. The 10 members independently agree on WT
to ±0.005, which is what a correctly mapped ensemble of separately trained
models should look like.

Note that the remap gate alone cannot validate the alphabet — both sides would
share a wrong order — which is why the external anchors exist.

## Units, and what transfers

Their `creilov_processed.csv` has 167,530 rows, matching our
`creilov_combined.csv` exactly, and the wild-type `fitness` agrees to the digit
(4.023229753). **The target is the same log10 mean fluorescence we use**, so
`guidance.threshold = 4.1` transfers with no rescaling.

One caveat worth knowing when reading `evaluate` output: the ported oracle's
predictions are range-compressed relative to the measured labels — on a 4,000-row
sample it spans **[3.429, 4.226]** against our labels' **[2.686, 4.199]**. This
is ordinary regression-to-the-mean for an MSE-trained model on a distribution
with many dead variants. Consequences:

- Rank-based metrics (`predictor_oracle_spearman`) are unaffected.
- `oracle_score_mean` is not comparable to a mean of *measured* labels, only to
  other oracle scores.
- Because the predicted maximum (4.226) sits only just above the 4.1 threshold,
  success-rate-above-threshold is **conservative**. Treat it as a relative
  measure across libraries, not an absolute hit rate.

## Limitations

- **Clean sequences only.** The 20-wide checkpoint has no mask representation, so
  every ESM mask column carries zero weight. `evaluate` scores at
  `mask_fraction=0.0`, which is the only valid use. Do not score masked inputs.
- **Variance channel is not meaningful.** See design choice 5.
- **Storage is 1.68× their weights** (3993 vs. 2380 input columns), most of it
  structural zeros. Not worth a sparse representation for 64 MB.
- **The port is not re-verified at load time.** The gates run in the conversion
  script; `model.port.json` is the durable record.
