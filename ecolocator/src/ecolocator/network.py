import weakref

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from .settings import NETWORK_DEFAULTS as D


@tf.keras.utils.register_keras_serializable(name="euclid_loss")
def euclid_loss(y_true, y_pred):
    """Euclidean distance between 2-D coordinates (location)."""
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


@tf.keras.utils.register_keras_serializable(package="ecolocator", name="StopGradient")
class StopGradient(tf.keras.layers.Layer):
    """Identity in the forward pass. In training it passes no gradient back
    to its input; otherwise (prediction, attribution) gradients pass."""

    def call(self, inputs, training=None):
        return tf.stop_gradient(inputs) if training else inputs


def build_pgs_network(
    num_covs: int,
    geno_mean: np.ndarray,
    geno_sd: np.ndarray,
    score_l2: float,
    nlayers: int = D["nlayers"],
    width: int = D["width"],
    dropout_prop: float = D["dropout_prop"],
    input_dropout: float = D["input_dropout"],
    weight_decay: float = D["weight_decay"],
    head_width: int = D["head_width"],
    loc_weight: float = D["loc_weight"],
    env_weight: float = D["env_weight"],
) -> tf.keras.Model:
    """The EcoLocator network.

    Genotypes (0/1/2 counts) are standardized inside the model with the
    training samples' means and SDs (geno_mean, geno_sd; one per SNP). Two
    routes follow:

    - location: input dropout, ``nlayers`` dense ELU layers of ``width``
      units with dropout after each, and a linear (x, y) head;
    - environment: for each covariate j, a linear score s_j of all SNPs (no
      bias, L2 penalty ``score_l2``) and a correction head h_j (``head_width``
      ELU units, then one linear unit) that reads s_j and the predicted
      location. The covariate is s_j + h_j, so the score enters with a fixed
      coefficient of 1 and its penalty acts on the SNP coefficients
      themselves.

    The predicted location enters h_j through a gradient stop that acts only
    in training: the covariate loss trains only the scores and the
    correction heads, and only the location loss trains the trunk. Outside
    training the gradient passes, so gradient-based attribution (SHAP) of a
    covariate includes the path through the predicted location.

    The penalties (``score_l2``, and ``weight_decay`` on the trunk and the
    heads) are multiplied by the loss weight of their output, so the loss
    weights set only the balance between location and covariates. The
    covariate loss is the mean squared error over the K covariates, so each
    covariate's error enters it with weight 1/K while its score keeps the
    full penalty: relative to that covariate's own error, the penalty is
    K * score_l2. The default penalty grid is tuned with this scaling.
    """
    loc_reg = tf.keras.regularizers.l2(weight_decay * loc_weight)
    env_reg = tf.keras.regularizers.l2(weight_decay * env_weight)
    geno_sd = np.where(np.asarray(geno_sd) > 0, geno_sd, 1.0)
    geno_input = tf.keras.Input(shape=(len(geno_mean),), name="geno_input")
    z = tf.keras.layers.Normalization(
        mean=np.asarray(geno_mean, dtype=np.float32),
        variance=np.square(geno_sd).astype(np.float32),
        name="standardize",
    )(geno_input)

    h = tf.keras.layers.Dropout(input_dropout)(z)
    for _ in range(nlayers):
        h = tf.keras.layers.Dense(width, activation="elu", kernel_regularizer=loc_reg)(
            h
        )
        h = tf.keras.layers.Dropout(dropout_prop)(h)
    loc_output = tf.keras.layers.Dense(2, kernel_regularizer=loc_reg, name="loc")(h)

    if num_covs == 0:
        model = tf.keras.Model(inputs=geno_input, outputs=loc_output)
        model.compile(optimizer="Adam", loss=[euclid_loss], loss_weights=[loc_weight])
        return model

    location = StopGradient(name="stop_location")(loc_output)
    covariates = []
    # TODO: consider dividing the score penalty by num_covs, so that
    # score_l2 means the same penalty per covariate for any number of
    # covariates (see the docstring); the default grid would need retuning.
    for j in range(num_covs):
        score = tf.keras.layers.Dense(
            1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(score_l2 * env_weight),
            name=f"score_{j}",
        )(z)
        hidden = tf.keras.layers.Dense(
            head_width, activation="elu", kernel_regularizer=env_reg, name=f"head_{j}"
        )(tf.keras.layers.Concatenate()([score, location]))
        correction = tf.keras.layers.Dense(
            1, kernel_regularizer=env_reg, name=f"correction_{j}"
        )(hidden)
        covariates.append(
            tf.keras.layers.Add(name=f"covariate_{j}")([score, correction])
        )
    env_output = (
        covariates[0]
        if num_covs == 1
        else tf.keras.layers.Concatenate(name="covariates")(covariates)
    )
    model = tf.keras.Model(inputs=geno_input, outputs=[loc_output, env_output])
    model.compile(
        optimizer="Adam",
        loss=[euclid_loss, "mse"],
        loss_weights=[loc_weight, env_weight],
    )
    return model


# The attribution model of each network (see _attribution_model), built once
# and kept as long as the network lives.
_ATTRIBUTION_MODELS = weakref.WeakKeyDictionary()


def _attribution_model(model: tf.keras.Model) -> tf.keras.Model:
    """A model with two outputs, both the covariate output of an EcoLocator
    network, computed with the network's own layers in one pass.

    The first is the network's covariate output itself. In the second, the
    predicted location reaches the correction heads through an unconditional
    gradient stop, so its gradient leaves out the path through the trunk.
    Called with ``training=False`` (dropout off), the two outputs are equal
    and differ only in their gradients."""
    if model in _ATTRIBUTION_MODELS:
        return _ATTRIBUTION_MODELS[model]
    num_covs = sum(layer.name.startswith("score_") for layer in model.layers)
    location = tf.keras.layers.Lambda(tf.stop_gradient)(model.get_layer("loc").output)
    covariates = []
    for j in range(num_covs):
        score = model.get_layer(f"score_{j}").output
        hidden = model.get_layer(f"head_{j}")(
            tf.keras.layers.Concatenate()([score, location])
        )
        correction = model.get_layer(f"correction_{j}")(hidden)
        covariates.append(tf.keras.layers.Add()([score, correction]))
    score_path = (
        covariates[0] if num_covs == 1 else tf.keras.layers.Concatenate()(covariates)
    )
    both = tf.keras.Model(inputs=model.input, outputs=[model.outputs[1], score_path])
    _ATTRIBUTION_MODELS[model] = both
    return both


def path_attributions(
    model: tf.keras.Model,
    x: np.ndarray,
    background: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    batch_values: int = 4_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """SHAP values of the covariate output of an EcoLocator network
    (build_pgs_network) by expected gradients (the method of
    ``shap.GradientExplainer``), split by path.

    For sample i and covariate c, phi_ic = mean over k of
    (x_i - b_k) * grad f_c(b_k + a_k (x_i - b_k)), with b_k a random row of
    ``background`` and a_k drawn from U(0, 1). Gradients add over the paths
    from the input to the output, so with the same draws for both parts:

    - score: the path through the polygenic scores, that is, the score
      itself and the correction head's use of it;
    - location: the path through the trunk and the predicted location.

    score + location is the full SHAP value. Summed over SNPs, it tends to
    f_c(x_i) - mean_b f_c(b) as ``n_samples`` grows.

    x and background are genotype rows (0/1/2 counts, as the model takes
    them). The draws go through the network in batches of about
    ``batch_values`` values per array (the interpolation points and each
    gradient). Returns (score, location), each of shape (samples, SNPs,
    covariates), in the units of the model's covariate output."""
    if len(model.outputs) < 2:
        raise ValueError("the model has no covariate output to attribute")
    both = _attribution_model(model)
    num_covs = both.outputs[0].shape[-1]

    @tf.function(reduce_retracing=True)
    def gradients(points):
        # One covariate at a time; the column slices must be taken inside
        # the tape, or the tape cannot trace them back to the input.
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(points)
            f_full, f_score = both(points, training=False)
            parts = [(f_full[:, c], f_score[:, c]) for c in range(num_covs)]
        g_full = [tape.gradient(f, points) for f, _ in parts]
        g_score = [tape.gradient(d, points) for _, d in parts]
        return tf.stack(g_full), tf.stack(g_score)

    x = np.asarray(x, dtype=np.float32)
    background = np.asarray(background, dtype=np.float32)
    n, n_snps = x.shape
    out_score = np.empty((n, n_snps, num_covs), np.float32)
    out_location = np.empty((n, n_snps, num_covs), np.float32)
    # Interpolation points per batch: the points, their offsets from the
    # sample, and two gradients per covariate each take this many rows.
    points = max(1, batch_values // n_snps)
    per_batch = max(1, points // n_samples)  # samples, when all draws fit
    draws = min(n_samples, points)  # draws per batch otherwise
    for start in range(0, n, per_batch):
        xs = x[start : start + per_batch]
        sum_score = np.zeros((len(xs), n_snps, num_covs), np.float32)
        sum_location = np.zeros((len(xs), n_snps, num_covs), np.float32)
        for first in range(0, n_samples, draws):
            k = min(draws, n_samples - first)
            b = background[rng.integers(len(background), size=(len(xs), k))]
            a = rng.random((len(xs), k, 1), dtype=np.float32)
            diff = xs[:, None, :] - b
            g_full, g_score = gradients(tf.constant((b + a * diff).reshape(-1, n_snps)))
            for c in range(num_covs):
                gf = g_full[c].numpy().reshape(diff.shape)
                gs = g_score[c].numpy().reshape(diff.shape)
                sum_score[:, :, c] += (diff * gs).sum(axis=1)
                sum_location[:, :, c] += (diff * (gf - gs)).sum(axis=1)
        out_score[start : start + per_batch] = sum_score / n_samples
        out_location[start : start + per_batch] = sum_location / n_samples
    return out_score, out_location


def build_network(
    n_snps: int,
    num_covs: int,
    nlayers: int = 10,
    width: int = 256,
    dropout_prop: float = 0.25,
    loc_weight: float = 1.0,
    env_weight: float = 1.0,
) -> tf.keras.Model:
    """A dense network: a trunk of dense ELU layers with linear location
    and covariate heads. EcoLocator.load reads models saved with it, and it
    serves as a baseline for comparisons with build_pgs_network."""
    geno_input = tf.keras.Input(shape=(n_snps,), name="geno_input")
    trunk = tf.keras.layers.BatchNormalization()(geno_input)
    for _ in range(int(np.floor(nlayers / 2))):
        trunk = tf.keras.layers.Dense(width, activation="elu")(trunk)
    trunk = tf.keras.layers.Dropout(dropout_prop)(trunk)
    for _ in range(int(np.ceil(nlayers / 2))):
        trunk = tf.keras.layers.Dense(width, activation="elu")(trunk)
    loc_output = tf.keras.layers.Dense(2)(tf.keras.layers.Dense(2)(trunk))
    if num_covs > 0:
        env_output = tf.keras.layers.Dense(num_covs)(
            tf.keras.layers.Dense(num_covs)(trunk)
        )
        model = tf.keras.Model(inputs=geno_input, outputs=[loc_output, env_output])
        losses = [euclid_loss, "mse"]
        loss_weights = [loc_weight, env_weight]
    else:
        model = tf.keras.Model(inputs=geno_input, outputs=loc_output)
        losses = [euclid_loss]
        loss_weights = [loc_weight]
    model.compile(
        optimizer="Adam",
        loss=losses,
        loss_weights=loss_weights,
    )
    return model


def build_callbacks(
    patience: int = 100,
    verbose: int = 1,
) -> list:
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        restore_best_weights=True,
    )
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=int(patience / 6),
        verbose=verbose,
        mode="auto",
        min_delta=0,
        cooldown=0,
        min_lr=0,
    )
    return [earlystop, reducelr]


def train_network(
    model: tf.keras.Model,
    traingen: np.ndarray,
    testgen: np.ndarray,
    trainlocs: list | np.ndarray,
    testlocs: list | np.ndarray,
    max_epochs: int = 5000,
    batch_size: int = 32,
    patience: int = 100,
    verbose: int = 1,
) -> tf.keras.callbacks.History:
    callbacks = build_callbacks(patience=patience, verbose=verbose)
    history = model.fit(
        traingen,
        trainlocs,
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=verbose,
        validation_data=(testgen, testlocs),
        callbacks=callbacks,
    )
    return history
