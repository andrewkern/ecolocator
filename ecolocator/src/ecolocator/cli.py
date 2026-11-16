from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from .settings import NETWORK_DEFAULTS as D

# The default penalty grid as the command line takes it: "20,60,200,600".
SCORE_L2_DEFAULT = ",".join(f"{v:g}" for v in D["score_l2"])


app = typer.Typer(
    add_completion=False,
    help="EcoLocator: predict geographic origin and environmental covariates from genotype data.",
    no_args_is_help=True,
)

# Network options shared by train and loo; see EcoLocator for their meaning.
NLayers = Annotated[
    int, typer.Option(help="Number of hidden layers in the location trunk.")
]
Width = Annotated[int, typer.Option(help="Units per hidden layer of the trunk.")]
DropoutProp = Annotated[float, typer.Option(help="Dropout after each trunk layer.")]
InputDropout = Annotated[
    float, typer.Option(help="Dropout on the genotypes entering the trunk.")
]
WeightDecay = Annotated[
    float, typer.Option(help="L2 weight decay of the trunk and heads.")
]
HeadWidth = Annotated[
    int, typer.Option(help="Units of each covariate's correction head.")
]
ScoreL2 = Annotated[
    str,
    typer.Option(
        help="L2 penalty of the polygenic scores: one number, or comma-separated "
        "numbers to try (the one with the lowest covariate validation loss is kept)."
    ),
]
LocWeight = Annotated[float, typer.Option(help="Loss weight on the location head.")]
EnvWeight = Annotated[
    float, typer.Option(help="Loss weight on the environmental-covariate head.")
]


def _make_model(cov_transforms: Optional[str], score_l2: str, **settings):
    """An EcoLocator from the command-line options. score_l2 is a
    comma-separated list; one value fixes the penalty."""
    from .ecolocator_class import EcoLocator

    try:
        penalties = tuple(float(v) for v in score_l2.split(","))
    except ValueError:
        raise typer.BadParameter(
            f"{score_l2!r}: give one number or comma-separated numbers",
            param_hint="--score-l2",
        )
    if not all(p >= 0 for p in penalties):
        raise typer.BadParameter("penalties must be >= 0", param_hint="--score-l2")
    return EcoLocator(
        cov_transforms=[t.strip() for t in cov_transforms.split(",")]
        if cov_transforms
        else None,
        score_l2=penalties,
        **settings,
    )


@app.command()
def train(
    genotypes: Path = typer.Option(
        ...,
        "--genotypes",
        "-g",
        exists=True,
        readable=True,
        help="Genotype input: .vcf, .vcf.gz, .zarr, or dosage-matrix TSV.",
    ),
    sample_data: Path = typer.Option(
        ...,
        "--sample-data",
        "-s",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Sample metadata TSV with sampleID, x, y, and covariate columns. Use NA for x/y of samples to predict.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        file_okay=False,
        help="Directory to save the fitted model (model.keras, arrays.npz, params.json, location_route.keras).",
    ),
    nlayers: NLayers = D["nlayers"],
    width: Width = D["width"],
    dropout_prop: DropoutProp = D["dropout_prop"],
    input_dropout: InputDropout = D["input_dropout"],
    weight_decay: WeightDecay = D["weight_decay"],
    head_width: HeadWidth = D["head_width"],
    score_l2: ScoreL2 = SCORE_L2_DEFAULT,
    loc_weight: LocWeight = D["loc_weight"],
    env_weight: EnvWeight = D["env_weight"],
    cov_transforms: Optional[str] = typer.Option(
        None,
        help="Comma-separated per-covariate transforms (e.g. 'none,log,log'). Valid: none, log.",
    ),
    max_epochs: int = typer.Option(5000, help="Maximum training epochs."),
    patience: int = typer.Option(
        100, help="Early-stopping patience (epochs without val improvement)."
    ),
    batch_size: int = typer.Option(32, help="Minibatch size."),
    min_mac: int = typer.Option(2, help="Minimum minor-allele count to retain a site."),
    max_snps: Optional[int] = typer.Option(
        None, help="Randomly subsample to this many SNPs (default: keep all)."
    ),
    train_split: float = typer.Option(
        0.9, help="Fraction of known-location samples used for training."
    ),
    seed: Optional[int] = typer.Option(
        None, help="RNG seed for splits and SNP subsetting."
    ),
    verbose: int = typer.Option(
        1, help="Keras verbosity (0=silent, 1=batches, 2=epochs)."
    ),
    weight_folds: int = typer.Option(
        5,
        help="Folds for the covariate weights of the genomic offset; each costs one "
        "more network fit. Below 2: weights from the validation samples only.",
    ),
) -> None:
    """Train an EcoLocator model and save it to OUT."""
    model = _make_model(
        cov_transforms,
        score_l2,
        nlayers=nlayers,
        width=width,
        dropout_prop=dropout_prop,
        input_dropout=input_dropout,
        weight_decay=weight_decay,
        head_width=head_width,
        loc_weight=loc_weight,
        env_weight=env_weight,
    )
    model.fit(
        genotype_path=str(genotypes),
        sample_data_path=str(sample_data),
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        min_mac=min_mac,
        max_snps=max_snps,
        train_split=train_split,
        seed=seed,
        verbose=verbose,
        weight_folds=weight_folds,
    )
    model.save(str(out))
    typer.echo(f"Saved model to {out}")


@app.command()
def predict(
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        exists=True,
        file_okay=False,
        help="Directory containing a saved EcoLocator model (from `ecolocator train`).",
    ),
    genotypes: Path = typer.Option(
        ...,
        "--genotypes",
        "-g",
        exists=True,
        readable=True,
        help="Genotype input: .vcf, .vcf.gz, .zarr, or dosage-matrix TSV.",
    ),
    sample_data: Path = typer.Option(
        ...,
        "--sample-data",
        "-s",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Sample metadata TSV. Rows with missing x/y are predicted.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        dir_okay=False,
        help="Output TSV path for predictions.",
    ),
) -> None:
    """Predict locations and covariates for samples with unknown coordinates."""
    from .ecolocator_class import EcoLocator

    el = EcoLocator.load(str(model))
    result = el.predict(str(genotypes), str(sample_data))
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, sep="\t", index=False)
    typer.echo(f"Predicted {len(result)} samples → {out}")


@app.command()
def offset(
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        exists=True,
        file_okay=False,
        help="Directory containing a saved EcoLocator model (from `ecolocator train`).",
    ),
    genotypes: Path = typer.Option(
        ...,
        "--genotypes",
        "-g",
        exists=True,
        readable=True,
        help="Genotype input: .vcf, .vcf.gz, .zarr, or dosage-matrix TSV.",
    ),
    sample_data: Path = typer.Option(
        ...,
        "--sample-data",
        "-s",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Sample metadata TSV. Every sample gets an offset; samples the model "
        "trained on use their out-of-fold predictions.",
    ),
    sites: Path = typer.Option(
        ...,
        "--sites",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Sites TSV: a column site, then one column per covariate of the model.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        dir_okay=False,
        help="Output TSV path: sampleID, site, offset.",
    ),
    no_error_scaling: bool = typer.Option(
        False,
        "--no-error-scaling",
        help="Weight covariates by env_weights_ alone, not divided by their prediction errors.",
    ),
) -> None:
    """Genomic offsets of the samples to sites (gardens, future climates)."""
    import pandas as pd

    from .ecolocator_class import EcoLocator

    el = EcoLocator.load(str(model))
    site_table = pd.read_csv(sites, sep="\t").set_index("site")
    result = el.offset(
        str(genotypes),
        str(sample_data),
        site_table,
        scale_by_error=not no_error_scaling,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, sep="\t", index=False)
    typer.echo(
        f"Offsets of {result.sampleID.nunique()} samples to {len(site_table)} sites → {out}"
    )


@app.command()
def loo(
    genotypes: Path = typer.Option(
        ...,
        "--genotypes",
        "-g",
        exists=True,
        readable=True,
        help="Genotype input: .vcf, .vcf.gz, .zarr, or dosage-matrix TSV.",
    ),
    sample_data: Path = typer.Option(
        ...,
        "--sample-data",
        "-s",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Sample metadata TSV with sampleID, x, y, and covariate columns (all rows known).",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        dir_okay=False,
        help="Output TSV path for combined LOO predictions.",
    ),
    max_folds: Optional[int] = typer.Option(
        None,
        help="Stop after this many folds. Useful for smoke-testing before a full run.",
    ),
    fold_start: int = typer.Option(
        0,
        help="First fold to run (0-based). Use with --fold-stop to split one LOO run across jobs.",
    ),
    fold_stop: Optional[int] = typer.Option(
        None,
        help="Run folds up to, but not including, this index (default: all).",
    ),
    shap_out: Optional[Path] = typer.Option(
        None,
        dir_okay=False,
        help=(
            "Also compute SHAP values of each held-out sample from its fold's "
            "model, and save them to this .npz file (arrays: sample_ids, "
            "snp_ids, outputs, values[fold, snp, output])."
        ),
    ),
    shap_background_size: int = typer.Option(
        100, help="Training samples used as the SHAP background in each fold."
    ),
    nlayers: NLayers = D["nlayers"],
    width: Width = D["width"],
    dropout_prop: DropoutProp = D["dropout_prop"],
    input_dropout: InputDropout = D["input_dropout"],
    weight_decay: WeightDecay = D["weight_decay"],
    head_width: HeadWidth = D["head_width"],
    score_l2: ScoreL2 = SCORE_L2_DEFAULT,
    loc_weight: LocWeight = D["loc_weight"],
    env_weight: EnvWeight = D["env_weight"],
    cov_transforms: Optional[str] = typer.Option(
        None,
        help="Comma-separated per-covariate transforms (e.g. 'none,log,log'). Valid: none, log.",
    ),
    max_epochs: int = typer.Option(5000, help="Maximum training epochs."),
    patience: int = typer.Option(
        100, help="Early-stopping patience (epochs without val improvement)."
    ),
    batch_size: int = typer.Option(32, help="Minibatch size."),
    min_mac: int = typer.Option(2, help="Minimum minor-allele count to retain a site."),
    max_snps: Optional[int] = typer.Option(
        None, help="Randomly subsample to this many SNPs (default: keep all)."
    ),
    train_split: float = typer.Option(
        0.9, help="Fraction of known-location samples used for training."
    ),
    seed: Optional[int] = typer.Option(
        None, help="RNG seed for splits and SNP subsetting."
    ),
    verbose: int = typer.Option(
        0, help="Keras verbosity (0=silent, 1=batches, 2=epochs)."
    ),
) -> None:
    """Leave-one-out fit-and-predict over all samples with known coordinates."""
    el = _make_model(
        cov_transforms,
        score_l2,
        nlayers=nlayers,
        width=width,
        dropout_prop=dropout_prop,
        input_dropout=input_dropout,
        weight_decay=weight_decay,
        head_width=head_width,
        loc_weight=loc_weight,
        env_weight=env_weight,
    )
    result = el.fit_predict_loo(
        genotype_path=str(genotypes),
        sample_data_path=str(sample_data),
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        min_mac=min_mac,
        max_snps=max_snps,
        train_split=train_split,
        seed=seed,
        verbose=verbose,
        max_folds=max_folds,
        fold_start=fold_start,
        fold_stop=fold_stop,
        shap_background_size=shap_background_size if shap_out else None,
    )
    if set(result.columns) - {"sampleID", "x", "y"}:
        # The score penalty that each fold chose (rows are folds).
        result["score_l2"] = el.loo_score_l2_
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, sep="\t", index=False)
    if shap_out is not None:
        import numpy as np

        shap_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shap_out, **el.loo_shap_)
        typer.echo(f"Per-fold SHAP values → {shap_out}")
    typer.echo(f"LOO complete: {len(result)} predictions → {out}")


@app.command()
def attribute(
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        exists=True,
        file_okay=False,
        help="Directory containing a saved EcoLocator model (from `ecolocator train`).",
    ),
    genotypes: Path = typer.Option(
        ...,
        "--genotypes",
        "-g",
        exists=True,
        readable=True,
        help="Genotype input: .vcf, .vcf.gz, .zarr, or dosage-matrix TSV.",
    ),
    sample_data: Path = typer.Option(
        ...,
        "--sample-data",
        "-s",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Sample metadata TSV. Rows with missing x/y are attributed; remaining rows form the background.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        dir_okay=False,
        help="Output TSV path for mean absolute SHAP summary (rows=SNPs, cols=output variables).",
    ),
    background_size: int = typer.Option(
        100,
        help="Number of training samples to use as SHAP background. Larger values are slower but more stable.",
    ),
    min_maf: Optional[float] = typer.Option(
        None,
        help="Exclude SNPs below this minor allele frequency from the output.",
    ),
    save_raw: bool = typer.Option(
        False,
        "--save-raw",
        help="Also save the per-sample raw SHAP values to <out-stem>_raw.tsv.",
    ),
    seed: Optional[int] = typer.Option(None, help="RNG seed for background sampling."),
) -> None:
    """Attribute SNP importance for samples with unknown coordinates using SHAP."""
    from .ecolocator_class import EcoLocator

    el = EcoLocator.load(str(model))
    out.parent.mkdir(parents=True, exist_ok=True)

    result = el.shap_values(
        genotype_path=str(genotypes),
        sample_data_path=str(sample_data),
        train_genotype_path=str(genotypes),
        train_sample_data_path=str(sample_data),
        background_size=background_size,
        min_maf=min_maf,
        seed=seed,
        raw=False,
    )
    result.to_csv(out, sep="\t", index=False)
    typer.echo(f"Attributed {len(result)} SNPs → {out}")

    if save_raw:
        raw_result = el.shap_values(
            genotype_path=str(genotypes),
            sample_data_path=str(sample_data),
            train_genotype_path=str(genotypes),
            train_sample_data_path=str(sample_data),
            background_size=background_size,
            min_maf=min_maf,
            seed=seed,
            raw=True,
        )
        raw_out = out.with_stem(out.stem + "_raw")
        raw_result.to_csv(raw_out, sep="\t", index=False)
        typer.echo(f"Raw SHAP values → {raw_out}")


if __name__ == "__main__":
    app()
