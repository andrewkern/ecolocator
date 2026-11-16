"""Default network settings of EcoLocator, shared by the class and the
command line. This module imports no TensorFlow, so the command line can
read the defaults without loading it."""

NETWORK_DEFAULTS = dict(
    nlayers=4,
    width=128,
    dropout_prop=0.25,
    input_dropout=0.1,
    weight_decay=2e-4,
    head_width=16,
    score_l2=(20.0, 60.0, 200.0, 600.0),
    loc_weight=1.0,
    env_weight=1.0,
)
