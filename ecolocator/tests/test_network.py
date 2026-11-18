import numpy as np
from ecolocator.network import build_network, build_callbacks


def test_build_network_dual_output():
    """testing that build_network returns a model with exactly two output heads"""
    model = build_network(n_snps=100, num_covs=3)
    assert len(model.outputs) == 2


def test_build_network_output_shape():
    """testing that build_network loc head outputs (n_samples, 2) and env head outputs (n_samples, num_covs)"""
    model = build_network(n_snps=100, num_covs=3)
    input = np.random.rand(5, 100)
    loc_out, env_out = model.predict(input, verbose=0)
    assert loc_out.shape == (5, 2)
    assert env_out.shape == (5, 3)


def test_build_callbacks_returns_two():
    """testing that build_callbacks returns a list of two callbacks"""
    callbacks = build_callbacks()
    assert len(callbacks) == 2


def test_build_callbacks_earlystopping_patience():
    """testing that patience parameter is correctly passed to EarlyStopping callback"""
    callbacks = build_callbacks(patience=42)
    earlystop = callbacks[0]
    assert earlystop.patience == 42


def test_build_network_forward_pass():
    """testing that build_network produces a model that runs a forward pass without error"""
    model = build_network(n_snps=50, num_covs=2)
    input = np.random.rand(3, 50)
    loc_out, env_out = model.predict(input, verbose=0)
    assert loc_out is not None
    assert env_out is not None


def _pgs(num_covs=2, n_snps=30, seed=0):
    from ecolocator.network import build_pgs_network

    rng = np.random.default_rng(seed)
    G = rng.integers(0, 3, size=(20, n_snps)).astype(np.float32)
    model = build_pgs_network(
        num_covs, G.mean(0), G.std(0), score_l2=20.0, nlayers=2, width=16
    )
    return model, G


def test_build_pgs_network_output_shapes():
    """testing that the PGS network outputs (n, 2) locations and (n, num_covs) covariates"""
    for num_covs in (1, 3):
        model, G = _pgs(num_covs=num_covs)
        loc_out, env_out = model.predict(G, verbose=0)
        assert loc_out.shape == (20, 2)
        assert env_out.shape == (20, num_covs)


def test_build_pgs_network_location_only():
    """testing that without covariates the PGS network has only the location output"""
    model, G = _pgs(num_covs=0)
    assert model.predict(G, verbose=0).shape == (20, 2)


def test_pgs_gradient_stop_only_in_training():
    """testing that the covariates' gradient reaches the trunk outside training but not in training"""
    import tensorflow as tf

    model, G = _pgs(num_covs=1)
    trunk = [w for w in model.trainable_weights if w.path.startswith("dense")]
    with tf.GradientTape(persistent=True) as tape:
        trained = tf.reduce_sum(model(G, training=True)[1])
        explained = tf.reduce_sum(model(G, training=False)[1])
    assert all(g is None for g in tape.gradient(trained, trunk))
    assert all(g is not None for g in tape.gradient(explained, trunk))


def test_pgs_save_load(tmp_path):
    """testing that the PGS network saves and loads with the same outputs"""
    import tensorflow as tf

    model, G = _pgs()
    model.save(tmp_path / "m.keras")
    loaded = tf.keras.models.load_model(tmp_path / "m.keras")
    np.testing.assert_allclose(
        loaded.predict(G, verbose=0)[1], model.predict(G, verbose=0)[1], atol=1e-6
    )


def test_attribution_model_matches_network_output():
    """testing that both outputs of the attribution model equal the network's covariate output"""
    from ecolocator.network import _attribution_model

    for num_covs in (1, 3):
        model, G = _pgs(num_covs=num_covs)
        full, score_path = _attribution_model(model)(G, training=False)
        expected = model(G, training=False)[1].numpy()
        np.testing.assert_allclose(full.numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(score_path.numpy(), expected, atol=1e-6)
        assert _attribution_model(model) is _attribution_model(model)


def test_path_attributions_are_complete():
    """testing that the two paths sum over SNPs to f(x) - f(b) with one background row"""
    from ecolocator.network import path_attributions

    model, G = _pgs(num_covs=2)
    score, location = path_attributions(
        model, G[:3], G[3:4], n_samples=2000, rng=np.random.default_rng(1)
    )
    assert score.shape == location.shape == (3, G.shape[1], 2)
    f = model(G, training=False)[1].numpy()
    np.testing.assert_allclose(
        (score + location).sum(axis=1), f[:3] - f[3], atol=0.02 * np.abs(f).max()
    )
    assert np.abs(location).sum() > 0


def test_path_attributions_without_location_input():
    """testing that the location path is zero when the heads ignore the location"""
    from ecolocator.network import path_attributions

    model, G = _pgs(num_covs=2)
    for c in range(2):
        head = model.get_layer(f"head_{c}")
        kernel, bias = head.get_weights()
        kernel[1:] = 0.0  # rows of the two location inputs
        head.set_weights([kernel, bias])
    score, location = path_attributions(
        model, G[:3], G[3:8], n_samples=50, rng=np.random.default_rng(1)
    )
    np.testing.assert_allclose(location, 0.0, atol=1e-6)
    assert np.abs(score).sum() > 0


def test_path_attributions_in_small_batches():
    """testing that splitting the draws of one sample into batches keeps completeness"""
    from ecolocator.network import path_attributions

    model, G = _pgs(num_covs=2)
    score, location = path_attributions(
        model,
        G[:2],
        G[3:4],
        n_samples=2000,
        rng=np.random.default_rng(1),
        batch_values=G.shape[1] * 300,
    )
    f = model(G, training=False)[1].numpy()
    np.testing.assert_allclose(
        (score + location).sum(axis=1), f[:2] - f[3], atol=0.02 * np.abs(f).max()
    )


def test_pgs_score_l2_penalty_independent_of_num_covs():
    """testing that a covariate's score penalty, relative to its data-loss
    weight, does not change with the number of covariates"""
    scaled_losses = {}
    for num_covs in (1, 3):
        model, G = _pgs(num_covs=num_covs, n_snps=10)
        model.get_layer("score_0").set_weights([np.ones((10, 1), dtype=np.float32)])
        model(G[:1])  # forward pass to populate layer.losses
        scaled_losses[num_covs] = (
            model.get_layer("score_0").losses[0].numpy() * num_covs
        )
    np.testing.assert_allclose(scaled_losses[1], scaled_losses[3])


def test_path_attributions_match_gradient_explainer():
    """testing that the two parts add up to shap.GradientExplainer's SHAP values"""
    import shap
    import tensorflow as tf
    from ecolocator.network import path_attributions

    model, G = _pgs(num_covs=2)
    x, background = G[:5], G[10:11]
    # One background row: both methods then estimate the same integral along
    # the line from it to each sample, and agree up to their random draws.
    score, location = path_attributions(
        model, x, background, n_samples=4000, rng=np.random.default_rng(1)
    )
    env = tf.keras.Model(inputs=model.input, outputs=model.outputs[1])
    expected = shap.GradientExplainer(env, background).shap_values(
        x, nsamples=4000, rseed=1
    )
    if isinstance(expected, list):  # older shap: one array per output
        expected = np.stack(expected, axis=-1)
    expected = np.asarray(expected).reshape(score.shape)
    np.testing.assert_allclose(
        score + location, expected, atol=0.02 * np.abs(expected).max()
    )
