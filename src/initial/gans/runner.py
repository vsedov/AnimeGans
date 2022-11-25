from src.initial.gans.data.dataset import set_up_dog_ds
from src.initial.gans.models.dcgan import shape_validation
from src.initial.gans.train import trainer


def setup():
    """
    Runs the DS Downloader, if you already have it,
    it should not need to download it.
    """
    set_up_dog_ds()
    shape_validation()
    trainer()
