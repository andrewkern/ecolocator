import numpy as np
import pandas as pd
import pytest

from conftest import structured_gea
from ecolocator.gea import (
    corrected_gea,
    env_effects,
    knots_for,
    location_basis,
    run_corrected_gea,
)


def test_knots_for_keeps_samples_per_basis_function():
    """testing that the knots shrink for small samples, down to the floor of 2"""
    assert knots_for(1000, 5) == 5
    assert knots_for(300, 5) == 3  # (3 + 2)^2 * 10 = 250 <= 300 < 490
    assert knots_for(50, 5) == 2


def test_location_basis_sums_to_one():
    """testing that the tensor-product B-splines have (k + 2)^2 columns and sum to 1"""
    xy = np.random.default_rng(1).uniform(0, 5, size=(100, 2))
    B = location_basis(xy, 4)
    assert B.shape == (100, 36)
    np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-10)


def test_env_effects_match_least_squares():
    """testing that the projected fit gives the full least-squares effect and SE"""
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 5, size=(200, 2))
    env = rng.normal(size=200)
    G = rng.binomial(2, 0.3, size=(200, 4)).astype(float)
    B = location_basis(xy, 3)
    beta, se = env_effects(G, env, B)
    # A full-rank design: the intercept and all but one spline column (the
    # splines sum to 1, so one is redundant with the intercept).
    X = np.column_stack([np.ones(200), env, B[:, 1:]])
    for j in range(4):
        coef, *_ = np.linalg.lstsq(X, G[:, j], rcond=None)
        resid = G[:, j] - X @ coef
        sigma2 = resid @ resid / (200 - X.shape[1])
        cov = sigma2 * np.linalg.inv(X.T @ X)
        np.testing.assert_allclose(beta[0, j], coef[1], rtol=1e-8)
        np.testing.assert_allclose(se[0, j], np.sqrt(cov[1, 1]), rtol=1e-6)


def test_corrected_gea_finds_causal_snps_beyond_structure():
    """testing that SNPs following the environment beyond location rank first and pass q < 0.05"""
    G, env, xy = structured_gea()
    results, summary = corrected_gea(G, pd.DataFrame({"env": env}), xy, min_maf=0.0)
    causal = results.snp_id >= 200
    assert results.loc[causal, "q"].max() < 0.05
    top = results.nlargest(5, "neglog10_p")
    assert set(top.snp_id) == set(range(200, 205))
    assert summary.loc[0, "n_snps"] == G.shape[1]
    assert np.isfinite(summary.loc[0, "gif"]) and summary.loc[0, "gif"] > 0


def test_corrected_gea_filters_and_missing_calls():
    """testing the MAF filter, mean imputation of missing calls, and the table columns"""
    G, env, xy = structured_gea(n=300, n_null=50, n_causal=0, seed=3)
    G[:, 0] = 0.0
    G[0, 0] = 1.0  # one copy: below min_mac 2
    G[:5, 1] = np.nan  # missing calls
    ids = [f"snp{j}" for j in range(G.shape[1])]
    results, summary = corrected_gea(
        G, pd.DataFrame({"a": env, "b": -env}), xy, snp_ids=ids, min_maf=0.0
    )
    assert "snp0" not in set(results.snp_id)
    assert results.loc[results.snp_id == "snp1", "p"].notna().all()
    assert list(results.columns) == [
        "snp_id",
        "env",
        "maf",
        "beta",
        "se",
        "z",
        "z_cal",
        "p",
        "neglog10_p",
        "q",
    ]
    assert list(summary.env) == ["a", "b"]
    # The variable's sign flips beta and z, not the test.
    a, b = results[results.env == "a"], results[results.env == "b"]
    np.testing.assert_allclose(a.z.to_numpy(), -b.z.to_numpy())
    np.testing.assert_allclose(a.p.to_numpy(), b.p.to_numpy())


def test_corrected_gea_checks_inputs():
    """testing that mismatched or non-finite inputs are refused"""
    G, env, xy = structured_gea(n=200, n_null=20, n_causal=0, seed=4)
    env = pd.DataFrame({"env": env})
    with pytest.raises(ValueError, match="one row per sample"):
        corrected_gea(G[:-1], env, xy)
    bad = xy.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="locations"):
        corrected_gea(G, env, bad)
    with pytest.raises(ValueError, match="min_mac"):
        corrected_gea(G, env, xy, min_maf=0.6)


def test_run_corrected_gea_from_files(gea_files, tmp_path):
    """testing the file-level run: sample matching and the default variables"""
    results, summary = run_corrected_gea(
        str(gea_files["genotypes"]),
        str(gea_files["samples"]),
        str(gea_files["locations"]),
        min_maf=0.0,
    )
    assert list(summary.env) == ["env"]  # x and y are not variables
    assert summary.loc[0, "n_samples"] == 300
    assert results.loc[results.snp_id.isin(["v30", "v31", "v32"]), "q"].max() < 0.05
    few = tmp_path / "few.tsv"
    pd.read_csv(gea_files["locations"], sep="\t").iloc[:10].to_csv(
        few, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="too few samples"):
        run_corrected_gea(
            str(gea_files["genotypes"]), str(gea_files["samples"]), str(few)
        )
