import pytest

import wandb


@pytest.mark.slow
def test_wandb_version():
    assert wandb.__version__ == "0.13.5"


@pytest.mark.slow
def test_quickstart():
    wandb.init(project="test_wandb")
    wandb.log({"loss": 0.5})
    wandb.finish()
