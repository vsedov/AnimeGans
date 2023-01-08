import builtins

from torchinfo import summary

from src.core import hc, hc_register_const, hp_register
from src.utils.constants import constants, constants_extra


def setup_constants():
    for name, const in constants().items():
        hc_register_const(name, const)


def setup_extra_constants():

    builtins.hc = hc
    for name, const in constants_extra().items():
        hc_register_const(name, const)


def setup_globals():

    @hp_register
    def to_default_device(*pointer):
        """Quick way to force default device to cuda"""
        for output in pointer:
            yield output.to(hc.DEFAULT_DEVICE)

    @hp_register
    def weights_init(m):
        """Quick Way to init the weights. Rather have this here, as there
        could be a case of repeating code.
        """
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    @hp_register 
    def get_core(param, default = None): 
        return hc.core.get(param, default)

    @hp_register
    def show_sum(network):
        """Show summary of a network"""
        summary(network)

    @hp_register
    def unpack(iterable):
        for _ in iterable:
            ...

    @hp_register
    def save(figure, file_name, title):
        #  TODO: (vsedov) (11:50:53 - 08/01/23): Refactor this, as the paths
        #  might be scuffed, which im not sure about
        file_path = f"{hc.DIR}reports/data/raw/"
        filename = f"{file_path}{file_name}"
        figure.suptitle(title)
        figure.savefig(f"{filename}.png")
