from src.core import hc

#  TODO: (vsedov) : Refactor this
if hc.initial["gan"]:
    from src.initial.gans.runner import setup

    setup()

if hc.initial["conv"]:
    from src.initial.image_proc.kernel_understanding import setup

    setup()
