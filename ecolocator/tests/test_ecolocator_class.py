import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from ecolocator import EcoLocator
from ecolocator.utils import replace_missing_data, sort_samples

# A small network with one fixed score penalty, so that each fit trains once.
# The penalty search has its own tests.
SMALL = dict(nlayers=2, width=32, score_l2=60.0)


def test_fit_returns_self(example_data):
    """testing that fit() returns the EcoLocator instance for method chaining"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    result = model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
    )
    assert result is model


def test_fit_sets_attributes(example_data):
    """testing that fit() sets expected attributes on the EcoLocator instance"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )
    assert hasattr(model, "model_")
    assert hasattr(model, "history_")
    assert hasattr(model, "cov_names_")
    assert hasattr(model, "means_")
    assert hasattr(model, "meanlong_")
    assert hasattr(model, "_kept_snp_indices_")
    assert hasattr(model, "seed_")
    assert model.num_covs_ == 3
    assert model.cov_names_ == ["cov1", "cov2", "cov3"]


def test_predict_returns_dataframe(example_data, tmp_path):
    """testing that predict() returns a DataFrame with correct columns"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    # creating a sample data file with one masked sample
    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    result = model.predict(str(matrix_path), str(masked_path))

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["sampleID", "x", "y", "cov1", "cov2", "cov3"]
    assert len(result) == 1


def test_predict_positive_with_log_transform(example_data, tmp_path):
    """predict() with log transform always returns positive covariate values"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL, cov_transforms=["none", "none", "log"])
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    result = model.predict(str(matrix_path), str(masked_path))

    assert result["cov3"].iloc[0] > 0


def test_fit_predict_loo_returns_all_samples(example_data):
    """fit_predict_loo() returns one prediction row per known sample"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    result = model.fit_predict_loo(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    sample_data = pd.read_csv(sample_data_path, sep="\t")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_data)
    assert "sampleID" in result.columns


def test_fit_predict_loo_max_folds(example_data):
    """fit_predict_loo() with max_folds returns exactly that many prediction rows"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    result = model.fit_predict_loo(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
        max_folds=2,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_fit_predict_loo_fold_range(example_data):
    """fold_start/fold_stop hold out exactly the samples in that fold range"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    result = model.fit_predict_loo(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
        seed=1,
        fold_start=2,
        fold_stop=4,
    )
    full_ids = model.fit_predict_loo(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=1,
        patience=1,
        train_split=0.6,
        min_mac=1,
        seed=1,
    )["sampleID"].tolist()

    assert result["sampleID"].tolist() == full_ids[2:4]


def test_fit_predict_loo_chunk_matches_full_run(example_data):
    """With a seed, a fold run alone gives the same prediction as in a full run"""
    _, _, matrix_path, sample_data_path = example_data
    kwargs = dict(max_epochs=5, patience=3, train_split=0.6, min_mac=1, seed=7)
    full = EcoLocator(**SMALL).fit_predict_loo(
        str(matrix_path), str(sample_data_path), fold_stop=3, **kwargs
    )
    chunk = EcoLocator(**SMALL).fit_predict_loo(
        str(matrix_path), str(sample_data_path), fold_start=2, fold_stop=3, **kwargs
    )

    pd.testing.assert_frame_equal(
        full.iloc[[2]].reset_index(drop=True), chunk.reset_index(drop=True)
    )


def test_fit_predict_loo_shap(example_data):
    """LOO with SHAP stores one SHAP row per fold and leaves predictions unchanged"""
    _, _, matrix_path, sample_data_path = example_data
    kwargs = dict(max_epochs=5, patience=3, train_split=0.6, min_mac=1, seed=5)
    plain = EcoLocator(**SMALL).fit_predict_loo(
        str(matrix_path), str(sample_data_path), fold_stop=3, **kwargs
    )
    model = EcoLocator(**SMALL)
    with_shap = model.fit_predict_loo(
        str(matrix_path),
        str(sample_data_path),
        fold_stop=3,
        shap_background_size=5,
        **kwargs,
    )

    pd.testing.assert_frame_equal(plain, with_shap)
    s = model.loo_shap_
    sample_data = pd.read_csv(sample_data_path, sep="\t")
    n_outputs = 2 + (sample_data.shape[1] - 3)
    assert list(s["sample_ids"]) == with_shap["sampleID"].tolist()
    assert list(s["outputs"][:2]) == ["x", "y"]
    assert s["values"].shape == (3, len(s["snp_ids"]), n_outputs)
    assert np.isfinite(s["values"]).all()
    assert all(s[k].dtype.kind == "U" for k in ("sample_ids", "snp_ids", "outputs"))


def test_fit_seed_is_reproducible(tmp_path, example_data):
    """fit() with the same seed gives the same predictions"""
    _, _, matrix_path, sample_data_path = example_data
    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)
    preds = []
    for _ in range(2):
        model = EcoLocator(**SMALL)
        model.fit(
            str(matrix_path),
            str(masked_path),
            max_epochs=5,
            patience=3,
            train_split=0.6,
            min_mac=1,
            seed=3,
            verbose=0,
        )
        preds.append(model.predict(str(matrix_path), str(masked_path)))

    pd.testing.assert_frame_equal(preds[0], preds[1])


def test_save_load_roundtrip(tmp_path, example_data):
    """save() and load() round-trip restores all attributes and hyperparameters"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    save_dir = str(tmp_path / "saved_model")
    model.save(save_dir)
    loaded = EcoLocator.load(save_dir)

    assert loaded.nlayers == model.nlayers
    assert loaded.cov_names_ == model.cov_names_
    assert np.isclose(loaded.meanlong_, model.meanlong_)
    assert np.isclose(loaded.meanlat_, model.meanlat_)
    np.testing.assert_array_equal(loaded._kept_snp_indices_, model._kept_snp_indices_)
    assert loaded.seed_ == model.seed_


def test_load_predict_matches(tmp_path, example_data):
    """a model loaded from disk produces a valid prediction DataFrame"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    save_dir = str(tmp_path / "saved_model")
    model.save(save_dir)
    loaded = EcoLocator.load(save_dir)

    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    result = loaded.predict(str(matrix_path), str(masked_path))

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["sampleID", "x", "y", "cov1", "cov2", "cov3"]


def test_shap_values_not_fitted_raises(example_data):
    """testing that shap_values() raises RuntimeError if called before fit()"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    with pytest.raises(RuntimeError):
        model.shap_values(
            str(matrix_path),
            str(sample_data_path),
            str(matrix_path),
            str(sample_data_path),
        )


def test_shap_values_returns_dataframe(example_data, tmp_path):
    """testing that shap_values() returns a summary DataFrame with one row per SNP"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    result = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
    )

    n_snps = len(model._kept_snp_indices_)
    expected_cols = ["snp_id", "x", "y"] + model.cov_names_
    assert isinstance(result, pd.DataFrame)
    assert len(result) == n_snps
    assert list(result.columns) == expected_cols
    assert set(result["snp_id"]) == {str(i) for i in model._kept_snp_indices_}


def test_shap_values_min_maf_filters_columns(example_data, tmp_path):
    """testing that shap_values() with min_maf filters out low-frequency SNPs"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    result_unfiltered = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
    )
    result_filtered = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
        min_maf=0.4,
    )

    assert len(result_filtered) < len(result_unfiltered)


def test_fit_no_covariates(no_covariate_data):
    """fit() works with x and y only — no covariate columns"""
    matrix_path, sample_data_path = no_covariate_data
    model = EcoLocator(**SMALL)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )
    assert model.num_covs_ == 0
    assert model.cov_names_ == []


def test_predict_no_covariates(no_covariate_data, tmp_path):
    """predict() with no covariates returns only sampleID, x, y columns"""
    matrix_path, sample_data_path = no_covariate_data
    model = EcoLocator(**SMALL)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )
    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)
    result = model.predict(str(matrix_path), str(masked_path))
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["sampleID", "x", "y"]


def test_fit_predict_loo_no_covariates(no_covariate_data):
    """fit_predict_loo() works with no covariates and returns only x, y columns"""
    matrix_path, sample_data_path = no_covariate_data
    model = EcoLocator(**SMALL)
    result = model.fit_predict_loo(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
        max_folds=2,
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["sampleID", "x", "y"]


def test_shap_values_no_covariates(no_covariate_data, tmp_path):
    """shap_values() with no covariates returns only snp_id, x, y columns"""
    matrix_path, sample_data_path = no_covariate_data
    model = EcoLocator(**SMALL)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )
    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)
    result = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["snp_id", "x", "y"]


def test_shap_values_raw_returns_wide_format(example_data, tmp_path):
    """shap_values() with raw=True returns one row per unknown sample with all SNP×variable columns"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    result = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
        raw=True,
    )

    n_snps = len(model._kept_snp_indices_)
    expected_n_cols = 1 + n_snps * (2 + len(model.cov_names_))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert "sampleID" in result.columns
    assert len(result.columns) == expected_n_cols


def test_shap_values_raw_no_covariates(no_covariate_data, tmp_path):
    """shap_values() raw=True with no covariates returns only _x and _y SHAP columns"""
    matrix_path, sample_data_path = no_covariate_data
    model = EcoLocator(**SMALL)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )
    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)
    result = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
        raw=True,
    )
    n_snps = len(model._kept_snp_indices_)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert "sampleID" in result.columns
    assert (
        len(result.columns) == 1 + n_snps * 2
    )  # sampleID + _x and _y per SNP, no env cols


def test_shap_values_raw_satisfies_additivity_location(example_data, tmp_path):
    """test that signed per-SNP SHAP values for the location head sum to (model output - baseline)"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
        seed=1,
    )

    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    raw = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
        raw=True,
    )
    assert len(raw) == 1

    train_genotypes, train_samples, train_snp_ids = model._get_genotypes(
        str(matrix_path)
    )
    train_genotypes = train_genotypes[model._kept_snp_indices_, :, :]
    train_ac = replace_missing_data(
        train_genotypes, rng=np.random.default_rng(model.seed_)
    )
    _, train_locs = sort_samples(train_samples, str(sample_data_path))
    known = np.argwhere(~np.isnan(train_locs[:, 0])).flatten()
    traingen = np.transpose(train_ac[:, known]).astype(np.float32)

    pred_genotypes, pred_samples, _ = model._get_genotypes(str(matrix_path))
    pred_genotypes = pred_genotypes[model._kept_snp_indices_, :, :]
    pred_ac = replace_missing_data(pred_genotypes)
    _, pred_locs = sort_samples(pred_samples, str(masked_path))
    unknown = np.argwhere(np.isnan(pred_locs[:, 0])).flatten()
    predgen = np.transpose(pred_ac[:, unknown]).astype(np.float32)

    snp_ids = train_snp_ids[model._kept_snp_indices_]

    model_loc = tf.keras.Model(
        inputs=model.model_.input, outputs=model.model_.output[0]
    )
    raw_loc = model_loc.predict(predgen, verbose=0)
    baseline_loc = model_loc.predict(traingen, verbose=0).mean(axis=0)

    row = raw.iloc[0]
    for out_idx, out_name in enumerate(["x", "y"]):
        shap_sum = sum(row[f"{snp_id}_{out_name}"] for snp_id in snp_ids)
        identity = shap_sum + baseline_loc[out_idx]
        assert np.isclose(identity, raw_loc[0, out_idx], atol=2.0)


def test_shap_values_raw_satisfies_additivity_env(example_data, tmp_path):
    """test that signed per-SNP SHAP values for the env head sum to (model output - baseline)"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
        seed=1,
    )

    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    raw = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
        raw=True,
    )
    assert len(raw) == 1

    train_genotypes, train_samples, train_snp_ids = model._get_genotypes(
        str(matrix_path)
    )
    train_genotypes = train_genotypes[model._kept_snp_indices_, :, :]
    train_ac = replace_missing_data(
        train_genotypes, rng=np.random.default_rng(model.seed_)
    )
    _, train_locs = sort_samples(train_samples, str(sample_data_path))
    known = np.argwhere(~np.isnan(train_locs[:, 0])).flatten()
    traingen = np.transpose(train_ac[:, known]).astype(np.float32)

    pred_genotypes, pred_samples, _ = model._get_genotypes(str(matrix_path))
    pred_genotypes = pred_genotypes[model._kept_snp_indices_, :, :]
    pred_ac = replace_missing_data(pred_genotypes)
    _, pred_locs = sort_samples(pred_samples, str(masked_path))
    unknown = np.argwhere(np.isnan(pred_locs[:, 0])).flatten()
    predgen = np.transpose(pred_ac[:, unknown]).astype(np.float32)

    snp_ids = train_snp_ids[model._kept_snp_indices_]

    model_env = tf.keras.Model(
        inputs=model.model_.input, outputs=model.model_.output[1]
    )
    raw_env = model_env.predict(predgen, verbose=0)
    baseline_env = model_env.predict(traingen, verbose=0).mean(axis=0)

    row = raw.iloc[0]
    for out_idx, cov in enumerate(model.cov_names_):
        shap_sum = sum(row[f"{snp_id}_{cov}"] for snp_id in snp_ids)
        identity = shap_sum + baseline_env[out_idx]
        assert np.isclose(identity, raw_env[0, out_idx], atol=2.0)


def test_fit_chooses_score_penalty(example_data):
    """testing that fit() tries every penalty of score_l2 and keeps the lowest validation loss"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**{**SMALL, "score_l2": (2.0, 200.0)})
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        seed=3,
    )
    assert set(model.score_l2_losses_) == {2.0, 200.0}
    assert model.score_l2_ == min(
        model.score_l2_losses_, key=model.score_l2_losses_.get
    )


def test_fit_fixed_score_penalty(example_data):
    """testing that one number for score_l2 skips the penalty search"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
    )
    assert list(model.score_l2_losses_) == [60.0]
    assert model.score_l2_ == 60.0


def test_fit_predict_loo_records_fold_penalties(example_data):
    """testing that LOO records the penalty that each fold chose"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**{**SMALL, "score_l2": (2.0, 200.0)})
    model.fit_predict_loo(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=1,
        patience=1,
        train_split=0.6,
        max_folds=2,
        seed=1,
    )
    assert len(model.loo_score_l2_) == 2
    assert set(model.loo_score_l2_) <= {2.0, 200.0}


def test_save_load_keeps_fitted_penalty(tmp_path, example_data):
    """testing that a saved and loaded model keeps the penalty that fit() chose"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=1,
        patience=1,
        train_split=0.6,
    )
    model.save(str(tmp_path / "m"))
    assert EcoLocator.load(str(tmp_path / "m")).score_l2_ == 60.0


def test_load_model_of_dense_network(tmp_path, example_data):
    """testing that a model saved with the dense network (build_network) loads and predicts"""
    import json

    from ecolocator.network import build_network

    _, _, matrix_path, sample_data_path = example_data
    fitted = EcoLocator(**SMALL)
    fitted.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=1,
        patience=1,
        train_split=0.6,
    )
    fitted.save(str(tmp_path / "m"))
    # Rewrite the saved model as one of the dense network: its model file,
    # and parameters with only its settings.
    build_network(
        n_snps=len(fitted._kept_snp_indices_),
        num_covs=fitted.num_covs_,
        nlayers=2,
        width=32,
    ).save(str(tmp_path / "m" / "model.keras"))
    params = json.loads((tmp_path / "m" / "params.json").read_text())
    old = {
        k: params[k]
        for k in (
            "seed",
            "cov_names",
            "num_covs",
            "transforms",
            "cov_transforms",
            "loc_weight",
            "env_weight",
        )
    }
    old.update(nlayers=2, width=32, dropout_prop=0.25)
    (tmp_path / "m" / "params.json").write_text(json.dumps(old))
    loaded = EcoLocator.load(str(tmp_path / "m"))
    assert loaded.score_l2_ is None
    sd = pd.read_csv(sample_data_path, sep="\t")
    sd.loc[:4, ["x", "y"]] = np.nan
    sd.to_csv(tmp_path / "pred_samples.tsv", sep="\t", index=False)
    assert (
        len(loaded.predict(str(matrix_path), str(tmp_path / "pred_samples.tsv"))) == 5
    )


def _masked(sample_data_path, tmp_path, n=2):
    sd = pd.read_csv(sample_data_path, sep="\t")
    sd.loc[: n - 1, ["x", "y"]] = np.nan
    path = tmp_path / "masked.tsv"
    sd.to_csv(path, sep="\t", index=False)
    return path


def test_fit_location_route_and_weights(example_data, tmp_path):
    """testing that fit() trains a location route and sets one non-negative weight per covariate"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL).fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=3,
        patience=2,
        train_split=0.6,
        seed=3,
    )
    assert model.location_route_ is not None
    assert model.env_weights_.shape == (3,)
    assert np.all(model.env_weights_ >= 0)
    assert model.env_mse_.shape == (3,)
    assert np.all(model.env_mse_ > 0)
    result = model.predict(
        str(matrix_path), str(_masked(sample_data_path, tmp_path)), location_route=True
    )
    assert list(result.columns) == [
        "sampleID",
        "x",
        "y",
        "cov1",
        "cov2",
        "cov3",
        "cov1_location",
        "cov2_location",
        "cov3_location",
    ]
    assert np.isfinite(result.drop(columns="sampleID").to_numpy()).all()


def test_offset(example_data_30, tmp_path):
    """testing that offset() gives every sample an offset from a model that did not see it"""
    _, _, matrix_path, sample_data_path = example_data_30
    masked = str(_masked(sample_data_path, tmp_path, n=5))
    model = EcoLocator(**SMALL).fit(
        str(matrix_path),
        masked,
        max_epochs=3,
        patience=2,
        train_split=0.6,
        seed=3,
        weight_folds=2,
    )
    sites = pd.DataFrame(
        {"cov1": [0.2, 0.8], "cov2": [0.5, 0.1], "cov3": [0.3, 0.9]}, index=["a", "b"]
    )
    off = model.offset(
        str(matrix_path), masked, sites, weights=[1.0, 0.0, 0.0], scale_by_error=False
    )
    assert list(off.columns) == ["sampleID", "site", "offset"]
    assert len(off) == 30 * len(sites)
    b = off[off.site == "b"].set_index("sampleID").offset
    # Samples with missing x/y: the fitted model's predictions.
    pred = model.predict(str(matrix_path), masked).set_index("sampleID")
    expected = np.abs(sites.loc["b", "cov1"] - pred.cov1) / model.sds_[0]
    np.testing.assert_allclose(b.loc[pred.index], expected, rtol=1e-5)
    # Training samples: their out-of-fold predictions.
    cv = model.crossfit_predictions_.set_index("sampleID")
    assert set(cv.index) == set(model.train_ids_)
    expected = np.abs(sites.loc["b", "cov1"] - cv.cov1) / model.sds_[0]
    np.testing.assert_allclose(b.loc[cv.index], expected, rtol=1e-5)
    # By default each weight is divided by the covariate's prediction error.
    scaled = model.offset(str(matrix_path), masked, sites, weights=[1.0, 0.0, 0.0])
    got = scaled[scaled.site == "b"].set_index("sampleID").offset.loc[pred.index]
    np.testing.assert_allclose(
        got, b.loc[pred.index] / np.sqrt(model.env_mse_[0]), rtol=1e-5
    )
    np.testing.assert_allclose(
        model.offset_weights_, model.env_weights_ / model.env_mse_
    )
    assert (model.offset(str(matrix_path), masked, sites).offset >= 0).all()


def test_offset_of_training_samples_needs_weight_folds(example_data, tmp_path):
    """testing that offsets of training samples need out-of-fold predictions"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL).fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=3,
        patience=2,
        train_split=0.6,
        seed=3,
        weight_folds=0,
    )
    sites = pd.DataFrame({"cov1": [0.2], "cov2": [0.5], "cov3": [0.3]}, index=["a"])
    with pytest.raises(ValueError, match="weight_folds"):
        model.offset(str(matrix_path), str(sample_data_path), sites)


def test_save_load_keeps_location_route(tmp_path, example_data_30):
    """testing that a saved and loaded model keeps its location route and weights"""
    _, _, matrix_path, sample_data_path = example_data_30
    model = EcoLocator(**SMALL).fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=3,
        patience=2,
        train_split=0.6,
        seed=3,
        weight_folds=2,
    )
    model.save(str(tmp_path / "m"))
    loaded = EcoLocator.load(str(tmp_path / "m"))
    np.testing.assert_allclose(loaded.env_weights_, model.env_weights_)
    np.testing.assert_allclose(loaded.env_mse_, model.env_mse_)
    pd.testing.assert_frame_equal(
        loaded.crossfit_predictions_, model.crossfit_predictions_
    )
    np.testing.assert_array_equal(loaded.train_ids_, model.train_ids_)
    masked = str(_masked(sample_data_path, tmp_path))
    a = model.predict(str(matrix_path), masked, location_route=True)
    b = loaded.predict(str(matrix_path), masked, location_route=True)
    np.testing.assert_allclose(
        a.drop(columns="sampleID").to_numpy(),
        b.drop(columns="sampleID").to_numpy(),
        rtol=1e-5,
    )


def test_fit_weights_from_validation_only(example_data):
    """testing that weight_folds below 2 gives weights from the validation samples, one per covariate"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(**SMALL).fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=3,
        patience=2,
        train_split=0.6,
        seed=3,
        weight_folds=0,
    )
    assert model.env_weights_.shape == (3,)
    assert np.all(model.env_weights_ >= 0)


def test_crossfit_weights_leave_the_model_unchanged(example_data_30, tmp_path, caplog):
    """testing that cross-fitted weights run without a fallback and do not change the fitted model's predictions"""
    _, _, matrix_path, sample_data_path = example_data_30
    masked = str(_masked(sample_data_path, tmp_path))
    preds = []
    for folds in (0, 2):
        model = EcoLocator(**SMALL).fit(
            str(matrix_path),
            str(sample_data_path),
            max_epochs=3,
            patience=2,
            train_split=0.6,
            seed=3,
            weight_folds=folds,
        )
        preds.append(model.predict(str(matrix_path), masked).drop(columns="sampleID"))
        assert np.all(model.env_weights_ >= 0)
    assert "Too few samples" not in caplog.text
    np.testing.assert_allclose(preds[0].to_numpy(), preds[1].to_numpy())


def test_offset_with_masked_covariates(example_data_30, tmp_path):
    """testing that samples to predict may have missing covariates as well as missing x/y"""
    _, _, matrix_path, sample_data_path = example_data_30
    sd = pd.read_csv(sample_data_path, sep="\t")
    sd.loc[:4, ["x", "y", "cov1", "cov2", "cov3"]] = np.nan
    masked = tmp_path / "masked_all.tsv"
    sd.to_csv(masked, sep="\t", index=False)
    model = EcoLocator(**SMALL).fit(
        str(matrix_path), str(masked), max_epochs=3, patience=2, train_split=0.6, seed=3
    )
    assert np.all(np.isfinite(model.env_weights_))
    sites = pd.DataFrame({"cov1": [0.2], "cov2": [0.5], "cov3": [0.3]}, index=["a"])
    off = model.offset(str(matrix_path), str(masked), sites)
    assert len(off) == 30
    assert np.isfinite(off.offset).all()
    pred = model.predict(str(matrix_path), str(masked), location_route=True)
    assert np.isfinite(pred.drop(columns="sampleID").to_numpy()).all()


def test_cli_offset(example_data_30, tmp_path):
    """testing that the offset command writes one offset per sample and site"""
    from typer.testing import CliRunner

    from ecolocator.cli import app

    _, _, matrix_path, sample_data_path = example_data_30
    model = EcoLocator(**SMALL).fit(
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
