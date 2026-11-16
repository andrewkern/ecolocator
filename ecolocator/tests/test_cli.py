import re

from typer.testing import CliRunner

from ecolocator.cli import app


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
