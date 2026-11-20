"""Location-corrected genotype-environment association (GEA).

For every SNP j and environmental variable e, the test fits by least squares

    g_j = b0 + b_e * env_e + f(x_hat, y_hat) + error

with g_j the genotype (0/1/2), env_e the environment at the sample's origin,
and f a smooth 2-D function of the sample's location as predicted from its
genotype by EcoLocator: a tensor product of cubic B-splines in x and y. The
statistic is ``z = b_e / SE(b_e)``. The z-scores of all tested SNPs are
calibrated by the genomic inflation factor, ``GIF = median(z^2) / 0.456``,
as in LFMM2: ``z_cal = z / sqrt(GIF)`` and ``p = 2 * Phi(-abs(z_cal))``.
q-values are Benjamini-Hochberg.

The predicted location is what the genotypes say about where a sample comes
from, so it summarizes population structure, the confounder of a GEA. The
true location would also carry the environment and remove its signal. Each
sample's location should come from a model that did not see it, for
example the leave-one-out predictions of ``EcoLocator.fit_predict_loo``
(``ecolocator loo``).

Population structure inflates the test more for common SNPs than for rare
ones, so the GIF depends on which SNPs are tested. It must come from the
SNPs whose results are used; ``min_maf`` sets that set.
"""

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, norm
from sklearn.preprocessing import SplineTransformer

from .utils import covariate_columns, filter_snps, read_genotypes

# median of the chi-square distribution with one degree of freedom
CHI2_MEDIAN = 0.456


def knots_for(n_samples: int, n_knots: int, min_per_term: int = 10) -> int:
    """The largest number of knots per axis, up to ``n_knots``, that keeps at
    least ``min_per_term`` samples per basis function. The tensor-product
    basis has (knots + 2)^2 functions; 2 knots (16 functions) is the floor."""
    k = n_knots
    while k > 2 and (k + 2) ** 2 * min_per_term > n_samples:
        k -= 1
    return k


def location_basis(locations: np.ndarray, n_knots: int) -> np.ndarray:
    """Tensor-product cubic B-spline basis of 2-D locations (samples x 2),
    with ``n_knots`` uniform knots per axis: samples x (n_knots + 2)^2."""
    splines = SplineTransformer(n_knots=n_knots, degree=3).fit_transform(
        np.asarray(locations, dtype=float)
    )
    bx, by = np.split(splines, 2, axis=1)  # x's splines, then y's
    return np.einsum("ni,nj->nij", bx, by).reshape(len(splines), -1)


def env_effects(genotypes: np.ndarray, env: np.ndarray, basis: np.ndarray):
    """Least-squares effect of each environmental variable (columns of
    ``env``, samples x variables) on every SNP (columns of ``genotypes``),
    adjusting for an intercept and ``basis``. Returns the effects and their
    standard errors, each variables x SNPs.

    The intercept and basis are projected out once for all variables: the
    projected variable is orthogonal to them, so its product with the
    genotypes equals its product with their residuals, and the residual sum
    of squares of each SNP follows from the projection of the genotypes."""
    env = np.asarray(env, dtype=float).reshape(len(genotypes), -1)
    # Orthonormal basis of the covariates' column space, cut at its numerical
    # rank: the B-splines sum to 1 in every row, so with an intercept the
    # design is rank-deficient.
    U, s, _ = np.linalg.svd(
        np.column_stack([np.ones(len(env)), basis]), full_matrices=False
    )
    Q = U[:, s > 1e-10 * s[0]]
    dof = len(env) - Q.shape[1] - 1
    if dof < 1:
        raise ValueError("too few samples for the location basis and the environment")
    QtG = Q.T @ genotypes
    rss = np.einsum("ij,ij->j", genotypes, genotypes) - np.einsum("ij,ij->j", QtG, QtG)
    e = env - Q @ (Q.T @ env)
    ee = np.einsum("ij,ij->j", e, e)[:, None]
    beta = (e.T @ genotypes) / ee
    resid_var = (rss - beta**2 * ee) / dof
    return beta, np.sqrt(np.maximum(resid_var, 1e-12) / ee)


def corrected_gea(
    genotypes: np.ndarray,
    env: pd.DataFrame,
    locations: np.ndarray,
    snp_ids=None,
    n_knots: int = 5,
    min_per_term: int = 10,
    min_mac: int = 2,
    min_maf: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The location-corrected GEA of every SNP against every environmental
    variable.

    ``genotypes``: samples x SNPs, 0/1/2 counts; NaN for missing calls, which
    are set to the SNP's mean. ``env``: one column per environmental
    variable, one row per sample. ``locations``: samples x 2, each sample's
    predicted location from a model that did not see it. ``snp_ids``: one
    name per SNP (default: column numbers). SNPs are tested when their
    minor-allele count is at least ``min_mac`` and their minor-allele
    frequency at least ``min_maf``; the GIF comes from those SNPs.

    Returns two tables:
      results  one row per SNP and variable: snp_id, env, maf, beta, se, z,
               z_cal, p, neglog10_p, q
      summary  one row per variable: env, n_samples, n_knots, n_snps, gif,
               n_q05, n_q10 (SNPs at q < 0.05 and q < 0.1)
    """
    G = np.asarray(genotypes, dtype=float)
    env = pd.DataFrame(env)
    xy = np.asarray(locations, dtype=float)
    if not (len(G) == len(env) == len(xy)):
        raise ValueError("genotypes, env, and locations need one row per sample each")
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("locations need two finite columns (x, y) for every sample")
    if not np.isfinite(env.to_numpy(dtype=float)).all():
        raise ValueError("env needs a finite value for every sample")
    snp_ids = np.arange(G.shape[1]) if snp_ids is None else np.asarray(snp_ids)

    missing = np.isnan(G)
    n_called = len(G) - missing.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(G, axis=0) / n_called
    maf = np.minimum(mean, 2 - mean) / 2
    keep = (2 * n_called * maf >= min_mac) & (maf >= min_maf)
    if not keep.any():
        raise ValueError("no SNP passes min_mac and min_maf")
    G, snp_ids, maf, mean = G[:, keep], snp_ids[keep], maf[keep], mean[keep]
    rows, cols = np.nonzero(missing[:, keep])
    G[rows, cols] = mean[cols]

    knots = knots_for(len(G), n_knots, min_per_term)
    beta, se = env_effects(G, env.to_numpy(dtype=float), location_basis(xy, knots))
    results, summary = [], []
    for name, b, s in zip(env.columns, beta, se):
        z = b / s
        gif = np.median(z**2) / CHI2_MEDIAN
        z_cal = z / np.sqrt(gif)
        p = 2 * norm.sf(np.abs(z_cal))
        q = false_discovery_control(p)
        results.append(
            pd.DataFrame(
                {
                    "snp_id": snp_ids,
                    "env": name,
                    "maf": maf,
                    "beta": b,
                    "se": s,
                    "z": z,
                    "z_cal": z_cal,
                    "p": p,
                    "neglog10_p": -(norm.logsf(np.abs(z_cal)) + np.log(2)) / np.log(10),
                    "q": q,
                }
            )
        )
        summary.append(
            {
                "env": name,
                "n_samples": len(G),
                "n_knots": knots,
                "n_snps": G.shape[1],
                "gif": gif,
                "n_q05": int((q < 0.05).sum()),
                "n_q10": int((q < 0.1).sum()),
            }
        )
    return pd.concat(results, ignore_index=True), pd.DataFrame(summary)


def run_corrected_gea(
    genotype_path: str,
    sample_data_path: str,
    locations_path: str,
    env_names=None,
    min_mac: int = 2,
    **settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """corrected_gea from files. ``genotype_path``: .vcf, .vcf.gz, .zarr, or a
    dosage-matrix TSV. ``sample_data_path``: TSV with sampleID and the
    environmental variables (``env_names``; default: every column but
    sampleID, x, and y). ``locations_path``: TSV with sampleID, x, and y, the
    predicted locations (for example the output of ``ecolocator loo``).

    Samples are used when they are in all three files and have a value of
    every variable and a location. SNPs pass ``utils.filter_snps`` (the
    biallelic sites and minimum minor-allele count of ``EcoLocator.fit``) on
    those samples, and a genotype is the count of allele 1, the encoding the
    network reads. ``settings`` go to corrected_gea."""
    genotypes, samples, snp_ids = read_genotypes(genotype_path)
    samples = np.asarray(samples, dtype=str)
    data = pd.read_csv(
        sample_data_path, sep="\t", index_col="sampleID", dtype={"sampleID": str}
    )
    locs = pd.read_csv(
        locations_path, sep="\t", index_col="sampleID", dtype={"sampleID": str}
    )
    names = covariate_columns(data.columns, env_names)
    env = data.reindex(samples)[names].astype(float)
    xy = locs.reindex(samples)[["x", "y"]].astype(float)
    use = (env.notna().all(axis=1) & xy.notna().all(axis=1)).to_numpy()
    if not use.any():
        raise ValueError("no sample has every variable and a location")

    genotypes, kept = filter_snps(genotypes[:, use], min_mac=min_mac)
    dosage = genotypes.to_allele_counts()[:, :, 1].T.astype(float)
    dosage[genotypes.is_missing().T] = np.nan
    return corrected_gea(
        dosage,
        env[use],
        xy.to_numpy()[use],
        snp_ids=np.asarray(snp_ids, dtype=str)[kept],
        min_mac=min_mac,
        **settings,
    )
