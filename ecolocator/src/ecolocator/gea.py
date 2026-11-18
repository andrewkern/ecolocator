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
from scipy.stats import norm
from sklearn.preprocessing import SplineTransformer

from .utils import read_genotypes

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
    locations = np.asarray(locations, dtype=float)
    bx = SplineTransformer(n_knots=n_knots, degree=3).fit_transform(locations[:, [0]])
    by = SplineTransformer(n_knots=n_knots, degree=3).fit_transform(locations[:, [1]])
    return np.einsum("ni,nj->nij", bx, by).reshape(len(locations), -1)


def _column_space(D: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis of the column space of D, cut at its numerical rank.
    The B-splines sum to 1 in every row, so with an intercept the design is
    rank-deficient; the SVD keeps only directions above rtol times the
    largest singular value."""
    U, s, _ = np.linalg.svd(D, full_matrices=False)
    return U[:, s > rtol * s[0]]


def env_effects(genotypes: np.ndarray, env: np.ndarray, basis: np.ndarray):
    """Least-squares effect of ``env`` on every column of ``genotypes``
    (samples x SNPs), adjusting for an intercept and ``basis``. Returns the
    effects and their standard errors, one per SNP."""
    Q = _column_space(np.column_stack([np.ones(len(env)), basis]))
    e = env - Q @ (Q.T @ env)
    R = genotypes - Q @ (Q.T @ genotypes)
    ee = e @ e
    beta = (e @ R) / ee
    dof = len(env) - Q.shape[1] - 1
    if dof < 1:
        raise ValueError("too few samples for the location basis and the environment")
    resid_var = ((R**2).sum(axis=0) - beta**2 * ee) / dof
    return beta, np.sqrt(np.maximum(resid_var, 1e-12) / ee)


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def read_dosages(genotype_path: str):
    """Genotypes of a .vcf, .vcf.gz, .zarr, or dosage-matrix file as
    alternate-allele counts: (samples x SNPs array, NaN for missing calls;
    sample IDs; SNP IDs)."""
    genotypes, samples, snp_ids = read_genotypes(genotype_path)
    counts = genotypes.to_n_alt(fill=-1).T.astype(float)
    counts[counts < 0] = np.nan
    return counts, np.asarray(samples, dtype=str), np.asarray(snp_ids, dtype=str)


def run_corrected_gea(
    genotype_path: str,
    sample_data_path: str,
    locations_path: str,
    env_names=None,
    **settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """corrected_gea from files. ``sample_data_path``: TSV with sampleID and
    the environmental variables (``env_names``; default: every column but
    sampleID, x, and y). ``locations_path``: TSV with sampleID, x, and y, the
    predicted locations (for example the output of ``ecolocator loo``).
    Samples are used when they are in all three files and have a value of
    every variable and a location. ``settings`` go to corrected_gea."""
    G, samples, snp_ids = read_dosages(genotype_path)
    data = pd.read_csv(sample_data_path, sep="\t", dtype={"sampleID": str}).set_index(
        "sampleID"
    )
    locs = pd.read_csv(locations_path, sep="\t", dtype={"sampleID": str}).set_index(
        "sampleID"
    )
    if env_names is None:
        env_names = [c for c in data.columns if c not in ("x", "y")]
    missing = [c for c in env_names if c not in data.columns]
    if missing:
        raise ValueError(f"not in the sample data: {', '.join(missing)}")
    env = data.reindex(samples)[list(env_names)].astype(float)
    xy = locs.reindex(samples)[["x", "y"]].astype(float)
    use = env.notna().all(axis=1).to_numpy() & xy.notna().all(axis=1).to_numpy()
    if not use.any():
        raise ValueError("no sample has every variable and a location")
    return corrected_gea(
        G[use], env[use], xy.to_numpy()[use], snp_ids=snp_ids, **settings
    )


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

    called = np.isfinite(G)
    mean = np.nanmean(np.where(called, G, np.nan), axis=0)
    G = np.where(called, G, mean)
    n_called = called.sum(axis=0)
    alt = (np.where(called, G, 0.0)).sum(axis=0)
    mac = np.minimum(alt, 2 * n_called - alt)
    with np.errstate(invalid="ignore", divide="ignore"):
        maf = mac / (2 * n_called)
    keep = (mac >= min_mac) & (maf >= min_maf)
    if not keep.any():
        raise ValueError("no SNP passes min_mac and min_maf")
    G, snp_ids, maf = G[:, keep], snp_ids[keep], maf[keep]

    knots = knots_for(len(G), n_knots, min_per_term)
    basis = location_basis(xy, knots)
    results, summary = [], []
    for name in env.columns:
        beta, se = env_effects(G, env[name].to_numpy(dtype=float), basis)
        z = beta / se
        gif = np.median(z**2) / CHI2_MEDIAN
        z_cal = z / np.sqrt(gif)
        p = 2 * norm.sf(np.abs(z_cal))
        q = bh_qvalues(p)
        results.append(
            pd.DataFrame(
                {
                    "snp_id": snp_ids,
                    "env": name,
                    "maf": maf,
                    "beta": beta,
                    "se": se,
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
