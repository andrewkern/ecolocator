# Python usage

This page shows the Python API. For the equivalent command-line workflow, see
{doc}`getting-started-cli`.

## Inputs

`EcoLocator` needs two inputs:

- **Genotypes** -- a `.vcf`, `.vcf.gz`, `.zarr`, or a dosage-matrix TSV
  (`sampleID` plus one column per SNP, entries `0`/`1`/`2`).
- **Sample metadata** -- a TSV with columns `sampleID`, `x`, `y`, and any number
  of environmental covariate columns. Use `NA` for the `x`, `y`, and covariate
  values of samples whose location and environment should be predicted.

## Train a model

```python
from ecolocator import EcoLocator

model = EcoLocator()
model.fit(
    genotype_path="./data/test_genotypes.vcf.gz",
    sample_data_path="./data/test_sample_data_cov.txt",
)
model.save("my_model")
```

`save()` writes a directory containing `model.keras`, `arrays.npz`, and
`params.json`.

### Network settings

`EcoLocator()` takes the network settings; the defaults are in
`ecolocator.settings.NETWORK_DEFAULTS`:

```python
model = EcoLocator(
    nlayers=4,          # dense ELU layers of the location trunk
    width=128,          # units per trunk layer
    dropout_prop=0.25,  # dropout after each trunk layer
    input_dropout=0.1,  # dropout on the genotypes
    weight_decay=2e-4,  # L2 decay of the trunk and the correction heads
    head_width=16,      # units of each covariate's correction head
    score_l2=(20, 60, 200, 600),  # L2 penalty of the polygenic scores
    loc_weight=1.0,     # loss weights: the balance between location
    env_weight=1.0,     # and the covariates
)
```

`score_l2` is one number (a fixed penalty) or a sequence of penalties:
`fit()` then fits the network once per penalty and keeps the fit with the
lowest covariate validation loss. The chosen penalty is `model.score_l2_`,
and each penalty's validation loss is in `model.score_l2_losses_`. The
covariate loss is the mean squared error over the K covariates, so each
covariate's error enters it with weight 1/K; each score's penalty is
divided by K as well, so `score_l2` is the same penalty, relative to a
covariate's own error, for any number of covariates.

`fit()` holds out `1 - train_split` of the samples with known coordinates
for validation (early stopping and the penalty choice), and raises an
error if that leaves no validation samples. Models saved with the earlier
dense network (`ecolocator.network.build_network`) load, predict, and
explain as they were fitted, but `fit()` does not refit them.

## Predict

```python
from ecolocator import EcoLocator

model = EcoLocator.load("my_model")
predictions = model.predict(
    genotype_path="./data/test_genotypes.vcf.gz",
    sample_data_path="./data/test_sample_data_cov.txt",
)
print(predictions)  # DataFrame: sampleID, x, y, <covariates...>
```


## Leave-one-out

`fit_predict_loo()` trains and predicts across folds, holding samples out in
turn:

```python
model = EcoLocator()
loo = model.fit_predict_loo(
    genotype_path="./data/test_genotypes.vcf.gz",
    sample_data_path="./data/test_sample_data_cov.txt",
)
```

`fold_start` and `fold_stop` run only a range of folds, so one LOO run can be
split across jobs; with `seed` set, a fold gives the same prediction in any
range. `shap_background_size` also computes the SHAP values of each held-out
sample from its fold's model and stores them in `model.loo_shap_`:

```python
loo = model.fit_predict_loo(
    genotype_path="./data/test_genotypes.vcf.gz",
    sample_data_path="./data/test_sample_data_cov.txt",
    fold_start=0,
    fold_stop=50,
    seed=1,
    shap_background_size=100,
)
model.loo_shap_["values"].shape  # (folds, SNPs, outputs)
```

Each fold searches the `score_l2` grid as `fit()` does; the penalty each
fold chose is in `model.loo_score_l2_`. For a cheaper LOO, fix `score_l2`
to one number, for example the penalty that `fit()` chose on all samples.

## SHAP feature attribution

```python
shap_out = model.shap_values(
    genotype_path="./data/test_genotypes.vcf.gz",
    sample_data_path="./data/test_sample_data_cov.txt",
    train_genotype_path="./data/test_genotypes.vcf.gz",
    train_sample_data_path="./data/test_sample_data_cov.txt",
)
```

The result has one row per SNP and one column per output (`x`, `y`, and
each covariate): the mean absolute SHAP value over the attributed samples.
Covariate values are in SDs of the covariate, after its transform.

`by_path=True` also splits each covariate's SHAP values by the network's two
paths. Every covariate `<cov>` gets two more columns: `<cov>_score`, the
part through the SNP's polygenic score (and the correction head's use of
it), and `<cov>_location`, the part through the predicted location:

```python
shap_out = model.shap_values(
    genotype_path="./data/test_genotypes.vcf.gz",
    sample_data_path="./data/test_sample_data_cov.txt",
    train_genotype_path="./data/test_genotypes.vcf.gz",
    train_sample_data_path="./data/test_sample_data_cov.txt",
    by_path=True,
)
shap_out[["snp_id", "cov1", "cov1_score", "cov1_location"]]
```

See {doc}`api/index` for the full parameter list of each method.
