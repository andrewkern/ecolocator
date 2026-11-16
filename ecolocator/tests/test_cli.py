import numpy as np
import pandas as pd
from typer.testing import CliRunner

from ecolocator import EcoLocator
from ecolocator.cli import app


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
    assert (tmp_path / "a_raw.tsv").exists()
