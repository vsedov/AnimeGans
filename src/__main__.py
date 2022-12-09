from src.core import hc

init = hc.initial

#  TODO: (vsedov) (10:46:41 - 08/12/22): remove this crap code
if init["train"]:
    if init["gan"]:
        from src.initial.gans.runner import setup
        setup()

elif init["test"]:
    from src.initial.gans.test import setup
    setup(check_point_amount=250, use_wandb=False, batch_size=64, dataload_amount=128)

if init["conv"]:
    from src.initial.image_proc.kernel_understanding import setup
    setup()
