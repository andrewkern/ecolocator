import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


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
    nlayers: int = 4,
    width: int = 128,
    dropout_prop: float = 0.25,
    input_dropout: float = 0.1,
    weight_decay: float = 2e-4,
    head_width: int = 16,
    loc_weight: float = 1.0,
    env_weight: float = 1.0,
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
    heads) are relative to the unweighted losses: each is multiplied by the
    loss weight of its output, so the loss weights set only the balance
    between location and covariates.
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


def build_location_route(
    num_covs: int,
    head_width: int = 16,
    weight_decay: float = 2e-4,
) -> tf.keras.Model:
    """The location route of the covariates: for each covariate j, a head
    g_j (``head_width`` ELU units, then one linear unit) that reads only a
    predicted location (x, y; normalized). It predicts what location alone
    tells about each covariate. EcoLocator.fit trains it after the network,
    so it does not change the network's fit."""
    location = tf.keras.Input(shape=(2,), name="location")
    reg = tf.keras.regularizers.l2(weight_decay)
    outputs = []
    for j in range(num_covs):
        hidden = tf.keras.layers.Dense(
            head_width, activation="elu", kernel_regularizer=reg, name=f"route_head_{j}"
        )(location)
        outputs.append(
            tf.keras.layers.Dense(1, kernel_regularizer=reg, name=f"route_{j}")(hidden)
        )
    output = (
        outputs[0]
        if num_covs == 1
        else tf.keras.layers.Concatenate(name="routes")(outputs)
    )
    model = tf.keras.Model(inputs=location, outputs=output)
    model.compile(optimizer="Adam", loss="mse")
    return model


def build_network(
    n_snps: int,
    num_covs: int,
    nlayers: int = 10,
    width: int = 256,
    dropout_prop: float = 0.25,
    loc_weight: float = 1.0,
    env_weight: float = 1.0,
) -> tf.keras.Model:
    """The network of the submitted manuscript (a dense trunk with linear
    location and covariate heads). EcoLocator now uses build_pgs_network;
    this one remains for comparisons with the submitted results."""
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
