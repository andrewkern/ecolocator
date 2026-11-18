from ecolocator import EcoLocator


def test_ecolocator_init_defaults():
    """EcoLocator initializes with correct default hyperparameters"""
    model = EcoLocator()
    assert model.nlayers == 4
    assert model.width == 128
    assert model.dropout_prop == 0.25
    assert model.input_dropout == 0.1
    assert model.weight_decay == 2e-4
    assert model.head_width == 16
    assert model.score_l2 == (20.0, 60.0, 200.0, 600.0)
    assert model.loc_weight == 1.0
    assert model.env_weight == 1.0
    assert model.cov_transforms is None


def test_ecolocator_init_custom():
    """EcoLocator stores custom hyperparameters correctly"""
    model = EcoLocator(nlayers=5, width=128, cov_transforms=["none", "log"])
    assert model.nlayers == 5
    assert model.width == 128
    assert model.cov_transforms == ["none", "log"]
