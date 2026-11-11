# CreiLOV S2 assay data

Original measurements: Chen et al. (2023), Supporting Data Sets S1 and S2:
https://doi.org/10.1021/acssynbio.2c00662

## Files and regeneration

The unmodified publisher workbooks are archived in `raw/`:

- `raw/sb2c00662_si_001.xlsx`: S1, 2,185 records including wild type.
- `raw/sb2c00662_si_002.xlsx`: S2, 165,428 records including wild type.

From the project root, regenerate `data/creilov.csv` with:

```powershell
uv run python -m creilov.data data
```

The default converter reads **S2 only** using the Python standard library.
S1 is handled by the separate export below. It preserves all 165,428 S2 records, their order,
and original numeric values. It does not modify either workbook.
Raw and generated data are ignored by Git; this documentation and converter
are versioned.

Original workbook SHA256 checksums:

```text
sb2c00662_si_001.xlsx 2ae429ec6a326270fa020a57703bf2a50cf05c2c21a3ab1c5d34045a58e25fe7
sb2c00662_si_002.xlsx 2312740855c03f3454b3cb61c767ea5ec3257fcf841f4970bc74d7f860afefdd
```

## Reference sequence and numbering

The reference is `CREILOV_WT` in ProteinGuide's `pdg/fitness/constants.py`:
https://github.com/junhaobearxiong/proteinguide/blob/main/pdg/fitness/constants.py

```text
MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA
```

This is 119 residues including the initiating methionine. S1 scans positions
2 through 119 (118 positions). All original amino acids in both workbooks were
checked against this reference. Positions are 1-based, so T7S changes Python
index 6. The label `wt` leaves the reference unchanged.

S2 has no premature-stop variants. All 165,428 reconstructed sequences have 119
residues. The mutation helper understands `Ter` as truncation before the stop
position, but the S2 dataset loader rejects stop variants or non-S2 rows.

## CSV columns

| Column | Meaning |
| --- | --- |
| `sequence` | Reconstructed 119-residue amino acid sequence |
| `mutations` | Original publisher label, e.g. `p.Thr7Ser` or comma-separated mutations |
| `n_mutations` | Number of mutation events in the label; not Hamming distance for stops |
| `library` | Always `S2` |
| `is_stop` | Always `0` for this library |
| `rep1`, `rep2`, `rep3` | Original replicate fluorescence estimates |
| `mean` | Supplied mean fluorescence |
| `log_rep1`, `log_rep2`, `log_rep3` | Supplied log10 replicate measurements |
| `fitness` | Supplied log10-of-mean fluorescence (`mean_log` in S2) |
| `source_file`, `source_row` | Original workbook and 1-based Excel row |

Higher fitness means brighter fluorescence. Fitness is not normalized to wild
type. The S2 wild-type fitness is 4.023229753. This table is a faithful conversion
of S2, not a claim to reproduce ProteinGuide's combined processed labels or
training splits. Column `fitness` is log10 of the arithmetic mean fluorescence,
not the average of the log-transformed replicates.

S2 contains 165,428 unique sequences: one wild type, 20 single mutants, and 165,407
variants with 2 through 15 mutations. No training/validation split is imposed.

## ProteinDataset usage

```python
from pathlib import Path
from creilov.data import load_dataset

dataset = load_dataset(Path("data/creilov.csv"))  # 165,428 S2 records
```

This returns ProteinGen's existing `ProteinDataset`, without a subclass.
`dataset.sequences` is a list of raw strings and `dataset.labels` is a float32
tensor of shape `[N]`. `dataset[i]` returns `{"sequence": ..., "labels": ...}`.
Analysis metadata remains in the CSV, not in model conditioning observations.
The dataset is constructed in memory from the CSV; no pickled object is needed.

The dataset itself does not tokenize, pad, mask, split, or train a model.
Its `collator(model, noise_fn, time_sampler)` creates a batch function for a
PyTorch DataLoader. The function tokenizes and pads with `model.tokenizer`,
copies the original tokens to `target_ids`, samples a masking time per sequence,
applies noise to `input_ids`, and stacks the fitness labels. Optional observations
are batched and passed through `model.preprocess_observations`.

For clean predictor inputs:

```python
from torch.utils.data import DataLoader
from proteingen.data import fully_unmasked, no_noise

# model is an already constructed ProteinGen model with a tokenizer.
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=dataset.collator(model, no_noise, fully_unmasked),
)
```

Each batch has `input_ids` and `target_ids` shaped `[B, L]`, `labels` shaped
`[B]`, and `observations={}` for this dataset. L includes tokenizer-specific
special tokens and padding. With no noise the two token arrays are equal.
For masking augmentation use `uniform_mask_noise(model.tokenizer)` and
`uniform_time`: positions are independently masked with probability `1 - t`.
The regression labels stay unchanged. `target_ids` are original sequence tokens,
not the regression targets; the latter are in `labels`.

## Separately aligned S1 export

```powershell
uv run python -m creilov.s1 data
```

This creates `creilov_s1_aligned.csv` and its provenance sidecar,
`creilov_s1_aligned.json`. It does not rewrite `creilov.csv`, either workbook,
or any training/benchmark configuration. The existing benchmark loader continues
to require S2-only data. Predictor training uses the separate combined pool below.

The alignment follows Yang et al., Appendix A.2:
https://arxiv.org/html/2505.15093#A2

```text
fitness_aligned = fitness_S1_raw + (fitness_S2_WT - fitness_S1_WT)
                = fitness_S1_raw - approximately 0.147130161
```

The exporter uses the original decimal strings so that the aligned S1 WT equals
the S2 CSV WT exactly. This shift applies only to the regression label, `fitness`.
`fitness_raw` stores the original S1 log10-of-mean value. The replicate columns,
`mean`, and `log_rep*` remain raw S1 measurements, not aligned values.

The CSV contains 2,122 full-length records: WT and 2,121 single substitutions.
The 63 premature-stop records remain in the raw workbook and are excluded from
this fixed-length model dataset. Of the exported records, 20 also occur in S2
(including WT); `overlaps_s2 = 1` flags these. A future merged dataset should
prefer S2 for shared sequences, leaving 2,102 additional S1-only sequences.
Do not randomly split duplicate sequences from the two libraries into different
partitions. Preserve the S2 split manifests when evaluating existing benchmarks.

Additional columns are `fitness_raw`, `log10_alignment_offset`, and `overlaps_s2`;
the other columns follow the table above, with `library = S1`. The JSON sidecar
records WT values, the exact offset, counts, input/output SHA256 checksums,
the method reference, and agreement on the 20 shared sequences. WT alignment
does not eliminate all assay differences: their residual RMSE is approximately
0.1402 log10 units after alignment.

## Combined predictor pool

```powershell
uv run python -m creilov.combined data
```

`creilov_combined.csv` contains the original 165,428 S2 sequences in their
original order, followed by 2,102 additional full-length S1 sequences using their
aligned `fitness` labels. S2 measurements win for the 20 shared sequences.
The two input CSVs are unchanged. `creilov_combined.json` records counts,
deduplication policy, and input/output SHA256 hashes. Alignment is checked
against the current S2 WT before combining.

Predictor training explicitly permits both libraries and uniformly samples its
configured training budget from the chosen split's training partition. This is
uniform per sequence, not a 50/50 library mixture. A 2,000-row draw has about
25 S1-only sequences on average; the actual fixed-seed selection is recorded in
the checkpoint metadata. Benchmark/oracle use the default S2-only loader and
reject the combined pool. Sequences are deduplicated before any partitioning.
