# Command-line usage

`EcoLocator` needs two inputs:

- **Genotypes** -- a `.vcf`, `.vcf.gz`, `.zarr`, or a dosage-matrix TSV
  (`sampleID` plus one column per SNP, entries `0`/`1`/`2`).
- **Sample metadata** -- a TSV with columns `sampleID`, `x`, `y`, and any number
  of environmental covariate columns. Use `NA` for the `x`, `y`, and covariate
  values of samples whose location and environment should be predicted.

Please find example data on Github at 
`./data/test_genotypes.vcf.gz` and `./data/test_sample_data_cov.txt`



Subcommands:

- `ecolocator train` — fit a model and save it to an output directory.
- `ecolocator predict` — predict locations and covariates for samples with unknown coordinates, using a saved model.
- `ecolocator loo` — leave-one-out cross-validation: iteratively hold out each sample, train, and predict.
- `ecolocator attribute` — attribute SNP importance for predicted samples using SHAP.

---

## `ecolocator train`

Trains an `EcoLocator` model on a genotype file plus a sample-metadata TSV and
writes the fitted model (`model.keras`, `arrays.npz`, `params.json`) to a
directory:

```bash
ecolocator train \
    --genotypes data/genotypes.tsv \
    --sample-data data/test_sample_data_cov.txt \
    --out out/my_model/
```

Genotype input may be `.vcf`, `.vcf.gz`, `.zarr`, or a dosage-matrix TSV. The
sample-metadata TSV must have columns `sampleID`, `x`, `y`, plus one column per
environmental covariate. Leave `x`/`y` blank (or `NA`) for samples whose
location is to be predicted.

Network flags: `--nlayers`, `--width`, `--dropout-prop`, `--input-dropout`,
`--weight-decay` (the location trunk), `--head-width` (each covariate's
correction head), `--score-l2` (the L2 penalty of the polygenic scores: one
number, or comma-separated numbers to try; default `20,60,200,600`),
`--loc-weight` and `--env-weight` (the balance between location and
covariates). Training flags: `--cov-transforms`
(comma-separated per-covariate, e.g. `none,log,log`), `--max-epochs`,
`--patience`, `--batch-size`, `--min-mac`, `--max-snps`, `--train-split`,
`--seed`, `--verbose`. Run `ecolocator train --help` for the full list.

---

## `ecolocator predict`

Loads a saved model and predicts locations and covariates for samples whose
`x`/`y` are missing in the sample-metadata file. Pass the **same masked
sample-data file** to both `train` and `predict` — `train` ignores the masked
rows during fitting, and `predict` returns predictions only for those rows:

```bash
ecolocator predict \
    --model out/my_model/ \
    --genotypes data/genotypes.tsv \
    --sample-data data/test_sample_data_cov.txt \
    --out out/predictions.tsv
```

Output is a tab-separated file with columns `sampleID`, `x`, `y`, and one
column per covariate.

---

## `ecolocator offset`

Genomic offsets of the samples to sites (common gardens, future climates),
from a model saved by `ecolocator train`:

```bash
ecolocator offset \
  --model my_model \
  --genotypes ./data/test_genotypes.vcf.gz \
  --sample-data ./data/test_sample_data_cov.txt \
  --sites sites.tsv \
  --out offsets.tsv
```

`sites.tsv` has a column `site` and one column per covariate of the model.
The output has one row per sample and site: `sampleID`, `site`, `offset`.
Every sample gets an offset; samples the model trained on use their
out-of-fold predictions, so the model must be trained with
`--weight-folds` of 2 or more (the default is 5). Each covariate is
weighted by how much the genotypes predict it beyond location, divided by
its prediction error; `--no-error-scaling` drops the division. See the
Python guide (Genomic offset) for the definition.

## `ecolocator loo`

Runs leave-one-out cross-validation over all samples. Pass the **full**
sample-metadata file (all coordinates known) — the method handles masking
internally, holding out one sample per fold, training a model, and predicting
that sample. The predictions are written when all folds are done:

```bash
ecolocator loo \
    --genotypes data/genotypes.tsv \
    --sample-data data/test_sample_data_cov.txt \
    --out out/loo_predictions.tsv
```

Use `--max-folds N` to run only the first N folds, useful for a quick
smoke-test before committing to a full run. All training hyperparameter flags
are identical to `train`.

### Splitting a LOO run across jobs

Fold *k* holds out the *k*-th sample (in genotype order) with known
coordinates. `--fold-start` and `--fold-stop` run only folds
`[fold-start, fold-stop)`, so a large LOO run can be split across many jobs
and the output tables concatenated afterwards:

```bash
ecolocator loo -g data/genotypes.tsv -s data/test_sample_data_cov.txt \
    --fold-start 0 --fold-stop 50 --seed 1 --out out/loo_chunk0.tsv
ecolocator loo -g data/genotypes.tsv -s data/test_sample_data_cov.txt \
    --fold-start 50 --fold-stop 100 --seed 1 --out out/loo_chunk1.tsv
```

With `--seed`, each fold's validation split and network initialization depend
only on the seed and the fold index, so a fold gives the same prediction in
any chunk. Results repeat exactly only with the same number of CPU threads.

Each fold searches the score penalty as `train` does, so a fold costs one fit
per penalty of `--score-l2`. To make LOO cheaper, give `--score-l2` one number,
for example the penalty `train` chose on all samples (`score_l2_fitted` in the
model's `params.json`).

### SHAP values from each fold

`--shap-out FILE.npz` also computes, in every fold, the SHAP values of the
held-out sample from the model that did not see it (background:
`--shap-background-size` training samples, default 100). The `.npz` file holds
`sample_ids`, `snp_ids`, `outputs` (`x`, `y`, then covariates), and `values`
with shape (folds, SNPs, outputs). The mean absolute value over folds is a
SNP-importance score computed entirely on held-out samples. The covariates'
SHAP values include the path through the predicted location, so they account
for everything the covariate prediction depends on.

---

## `ecolocator attribute`

Loads a saved model and computes SNP importance scores for samples with unknown
coordinates using SHAP. Pass the **same masked sample-data file** as used for
`predict` — rows with known coordinates form the background for SHAP, rows with
missing `x`/`y` are the samples being attributed:

```bash
ecolocator attribute \
    --model out/my_model/ \
    --genotypes data/genotypes.tsv \
    --sample-data data/test_sample_data_cov.txt \
    --out out/attributions.tsv
```

Output is a tab-separated file with one row per SNP and one column per output
variable (`x`, `y`, and each covariate), containing mean absolute SHAP values
across all attributed samples. The `snp_id` column identifies each SNP using
whatever real identifier was available in `--genotypes`: the VCF `ID` field
(falling back to `CHROM:POS` for every SNP if any are missing an ID, so
identifiers don't mix formats), the zarr store's equivalent fields, or the
genotype matrix's own column headers.

Use `--background-size` (default 100) to control how many training samples are
used as the SHAP background — larger values are more stable but slower. Use
`--min-maf` to exclude low-frequency SNPs from the output. Add `--save-raw` to
also write per-sample SHAP values to `<out-stem>_raw.tsv` alongside the summary.

