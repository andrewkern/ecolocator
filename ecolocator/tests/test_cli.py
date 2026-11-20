import re

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from ecolocator import EcoLocator
from ecolocator.cli import app


def _plain(output):
    """CLI output without the color escape codes that rich may add."""
    return re.sub(r"\x1b\[[0-9;]*m", "", output)


# A small, fixed-penalty network, so the CLI runs one fit per fold.
_SMALL_NETWORK_ARGS = [
    "--nlayers",
    "2",
    "--width",
    "32",
    "--score-l2",
    "60",
    "--train-split",
    "0.6",
    "--min-mac",
    "1",
    "--max-epochs",
    "3",
    "--patience",
    "2",
]


def test_attribute_by_path(example_data, tmp_path):
    """testing that `ecolocator attribute --by-path` splits every covariate"""
    _, _, matrix_path, sample_data_path = example_data
    EcoLocator(nlayers=2, width=32, score_l2=60.0).fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=2,
        patience=1,
        train_split=0.6,
        min_mac=1,
    ).save(str(tmp_path / "m"))
    sd = pd.read_csv(sample_data_path, sep="\t")
    sd.loc[[0, 1], ["x", "y"]] = np.nan
    sd.to_csv(tmp_path / "masked.tsv", sep="\t", index=False)
    args = ["attribute", "-m", str(tmp_path / "m"), "-g", str(matrix_path)]
    args += ["-s", str(tmp_path / "masked.tsv"), "--by-path", "--save-raw"]
    result = CliRunner().invoke(app, [*args, "-o", str(tmp_path / "a.tsv")])
    assert result.exit_code == 0, result.output
    table = pd.read_csv(tmp_path / "a.tsv", sep="\t")
    for c in ("cov1", "cov2", "cov3"):
        assert {c, f"{c}_score", f"{c}_location"} <= set(table)
    # The summary is the mean absolute value of the raw table.
    raw = pd.read_csv(tmp_path / "a_raw.tsv", sep="\t")
    snp = table.snp_id.iloc[0]
    for c in ("x", "cov1", "cov2_score", "cov3_location"):
        np.testing.assert_allclose(
            raw[f"{snp}_{c}"].abs().mean(), table[c].iloc[0], rtol=1e-5
        )


def test_bad_score_l2_is_a_usage_error(example_data, tmp_path):
    """testing that the CLI refuses an infinite penalty before it reads the data"""
    _, _, matrix_path, sample_data_path = example_data
    args = ["train", "-g", str(matrix_path), "-s", str(sample_data_path)]
    args += ["-o", str(tmp_path / "m"), "--score-l2", "inf"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    # Rich may color the message; drop the escape codes before searching.
    assert "score-l2" in _plain(result.output)
    assert not (tmp_path / "m").exists()


def test_loo_with_covariates_includes_fold_score_l2(example_data, tmp_path):
    """testing that `ecolocator loo` adds a fold_score_l2 column when there are covariates"""
    _, _, matrix_path, sample_data_path = example_data
    out_path = tmp_path / "loo.tsv"
    args = ["loo", "-g", str(matrix_path), "-s", str(sample_data_path)]
    args += ["-o", str(out_path), "--max-folds", "2"]
    args += _SMALL_NETWORK_ARGS
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    out = pd.read_csv(out_path, sep="\t")
    assert "fold_score_l2" in out.columns
    assert len(out) == 2


def test_loo_without_covariates_omits_fold_score_l2(no_covariate_data, tmp_path):
    """testing that `ecolocator loo` omits fold_score_l2 when there are no covariates"""
    matrix_path, sample_data_path = no_covariate_data
    out_path = tmp_path / "loo.tsv"
    args = ["loo", "-g", str(matrix_path), "-s", str(sample_data_path)]
    args += ["-o", str(out_path), "--max-folds", "2"]
    args += _SMALL_NETWORK_ARGS
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    out = pd.read_csv(out_path, sep="\t")
    assert "fold_score_l2" not in out.columns
    assert len(out) == 2


def test_train_saves_a_loadable_model(example_data, tmp_path):
    """testing that `ecolocator train` saves a model that loads and predicts"""
    _, _, matrix_path, sample_data_path = example_data
    out_path = tmp_path / "m"
    args = ["train", "-g", str(matrix_path), "-s", str(sample_data_path)]
    args += ["-o", str(out_path)]
    args += _SMALL_NETWORK_ARGS
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert (out_path / "model.keras").exists()
    assert (out_path / "params.json").exists()
    assert (out_path / "arrays.npz").exists()
    sd = pd.read_csv(sample_data_path, sep="\t")
    sd.loc[[0, 1], ["x", "y"]] = np.nan
    sd.to_csv(tmp_path / "masked.tsv", sep="\t", index=False)
    pred = EcoLocator.load(str(out_path)).predict(
        str(matrix_path), str(tmp_path / "masked.tsv")
    )
    assert len(pred) == 2


def test_score_l2_trailing_comma_is_a_usage_error(example_data, tmp_path):
    """testing that a trailing comma in --score-l2 is a clean usage error, not a crash"""
    _, _, matrix_path, sample_data_path = example_data
    args = ["train", "-g", str(matrix_path), "-s", str(sample_data_path)]
    args += ["-o", str(tmp_path / "m"), "--score-l2", "20,60,"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    assert "score-l2" in _plain(result.output)
    assert not (tmp_path / "m").exists()


def test_offset_writes_one_row_per_sample_and_site(example_data_30, tmp_path):
    """testing that `ecolocator offset` writes one offset per sample and site"""
    _, _, matrix_path, sample_data_path = example_data_30
    model = EcoLocator(nlayers=2, width=32, score_l2=60.0).fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=3,
        patience=2,
        train_split=0.6,
        seed=3,
        weight_folds=2,
    )
    model.save(str(tmp_path / "m"))
    pd.DataFrame(
        {"site": ["a", "b"], "cov1": [0.2, 0.8], "cov2": [0.5, 0.1], "cov3": [0.3, 0.9]}
    ).to_csv(tmp_path / "sites.tsv", sep="\t", index=False)
    out = tmp_path / "offsets.tsv"
    result = CliRunner().invoke(
        app,
        [
            "offset",
            "--model",
            str(tmp_path / "m"),
            "--genotypes",
            str(matrix_path),
            "--sample-data",
            str(sample_data_path),
            "--sites",
            str(tmp_path / "sites.tsv"),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    off = pd.read_csv(out, sep="\t")
    assert len(off) == 60
    assert set(off.site) == {"a", "b"}


def test_gea_writes_results_and_summary(gea_files, tmp_path):
    """testing that `ecolocator gea` writes one row per SNP and a summary per variable"""

    def run(env):
        return CliRunner().invoke(
            app,
            [
                "gea",
                "-g",
                str(gea_files["genotypes"]),
                "-s",
                str(gea_files["samples"]),
                "-l",
                str(gea_files["locations"]),
                "-o",
                str(tmp_path / "gea.tsv"),
                "--env",
                env,
            ],
        )

    result = run("env")
    assert result.exit_code == 0, result.output
    out = pd.read_csv(tmp_path / "gea.tsv", sep="\t")
    assert set(out.env) == {"env"}
    summary = pd.read_csv(tmp_path / "gea_summary.tsv", sep="\t")
    assert list(summary.env) == ["env"]
    bad = run("nope")
    assert bad.exit_code == 2
    assert "nope" in _plain(bad.output)
