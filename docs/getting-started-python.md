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
`params.json`, and, for a model with covariates, `location_route.keras`
(see Genomic offset).

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

## Genomic offset

After the network, `fit()` trains a location route: for each covariate, a
small head that predicts it from the predicted location alone.
`model.env_weights_` holds how much of each covariate the network predicts
beyond its location route, R²(c ~ route + prediction) − R²(c ~ route),
floored at 0, from cross-fitted predictions of all samples with known
coordinates: `fit(..., weight_folds=5)` fits the network and its route on
4/5 of the samples and predicts the rest, five times. This costs five more
network fits; `weight_folds=0` uses the validation samples of the main fit
only, which is cheaper but noisy when there are few of them. A covariate that varies with location but
not with the genotypes beyond location gets a weight near 0; a covariate
under selection keeps a positive weight from its polygenic score.

`model.env_mse_` holds each covariate's cross-fitted prediction error
(mean squared error in normalized units), from the same folds.

`offset()` gives the genomic offset of every sample in the sample data to
each site (for example a common garden or a future climate): the Euclidean
distance between the covariates EcoLocator predicts for the sample and
those of the site, in normalized units, with each covariate weighted by
`env_weights_ / env_mse_` (`model.offset_weights_`). A covariate counts
more when the genotype predicts it beyond location and when the network
predicts it well. `weights=` replaces `env_weights_`;
`scale_by_error=False` drops the division by `env_mse_`.

Every prediction comes from a model that did not see the sample. Samples
that `fit()` trained on (for example trees of known origin) use their
out-of-fold predictions from the weight folds
(`model.crossfit_predictions_`); all other samples use the fitted model.
Offsets of training samples therefore need `weight_folds` of 2 or more.

```python
import pandas as pd

# One row per site, a column for every covariate of the model.
sites = pd.DataFrame(
    {"cov1": [0.2, 0.8], "cov2": [0.5, 0.1], "cov3": [0.3, 0.9]},
    index=["garden_a", "garden_b"],
)
offsets = model.offset(
    genotype_path="./data/test_genotypes.vcf.gz",
    sample_data_path="./data/test_sample_data_cov.txt",
    sites=sites,
)  # DataFrame: sampleID, site, offset
```

`predict(..., location_route=True)` adds the location-route prediction of
each covariate (columns `<covariate>_location`).


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

## SHAP feature attribution

```python
shap_out = model.shap_values(
    genotype_path="./data/test_genotypes.vcf.gz",
    sample_data_path="./data/test_sample_data_cov.txt",
    train_genotype_path="./data/test_genotypes.vcf.gz",
    train_sample_data_path="./data/test_sample_data_cov.txt",
)
```

See {doc}`api/index` for the full parameter list of each method.
