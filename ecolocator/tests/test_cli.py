import re

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from ecolocator import EcoLocator
from ecolocator.cli import app

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


def test_bad_score_l2_is_a_usage_error(example_data, tmp_path):
    """testing that the CLI refuses an infinite penalty before it reads the data"""
    _, _, matrix_path, sample_data_path = example_data
    args = ["train", "-g", str(matrix_path), "-s", str(sample_data_path)]
    args += ["-o", str(tmp_path / "m"), "--score-l2", "inf"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    # Rich may color the message; drop the escape codes before searching.
    assert "score-l2" in re.sub(r"\x1b\[[0-9;]*m", "", result.output)
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
    assert "score-l2" in re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert not (tmp_path / "m").exists()
