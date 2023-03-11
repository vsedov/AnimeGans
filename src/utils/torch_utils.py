import numpy as np
import torch
from torchvision import utils as vutils

"""Hair and eye color mappings and dictionaries."""
hair_mapping = [
    "orange",
    "white",
    "aqua",
    "gray",
    "green",
    "red",
    "purple",
    "pink",
    "blue",
    "black",
    "brown",
    "blonde",
]
"""Hair color mapping."""
hair_dict = {
    "orange": 0,
    "white": 1,
    "aqua": 2,
    "gray": 3,
    "green": 4,
    "red": 5,
    "purple": 6,
    "pink": 7,
    "blue": 8,
    "black": 9,
    "brown": 10,
    "blonde": 11,
}

"""Eye color mapping."""
eye_mapping = [
    "gray",
    "black",
    "orange",
    "pink",
    "yellow",
    "aqua",
    "purple",
    "green",
    "brown",
    "red",
    "blue",
]
"""Eye color mapping."""
eye_dict = {
    "gray": 0,
    "black": 1,
    "orange": 2,
    "pink": 3,
    "yellow": 4,
    "aqua": 5,
    "purple": 6,
    "green": 7,
    "brown": 8,
    "red": 9,
    "blue": 10,
}


def save_model(model, optimizer, step, file_path):
    """Save a PyTorch model checkpoint.

    This function saves the model's state_dict, optimizer's state_dict, and current step to a file.

    Args:
        model (nn.Module): the PyTorch model to be saved.
        optimizer (torch.optim): the optimizer used for training the model.
        step (int): the current training step.
        file_path (str): the path to the file where the checkpoint will be saved.

    Returns:
        None
    """
    state = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "step": step,
    }
    torch.save(state, file_path)


def load_model(model, optimizer, file_path):
    """Load a PyTorch model checkpoint.

    This function loads the state_dict, optimizer's state_dict, and current step from a file and updates the model and optimizer accordingly.

    Args:
        model (nn.Module): the PyTorch model to be updated.
        optimizer (torch.optim): the optimizer used for training the model.
        file_path (str): the path to the file where the checkpoint is saved.

    Returns:
        model (nn.Module): the updated model.
        optimizer (torch.optim): the updated optimizer.
        start_step (int): the step from which training should resume.
    """
    prev_state = torch.load(file_path)

    model.load_state_dict(prev_state["model"])
    optimizer.load_state_dict(prev_state["optim"])
    start_step = prev_state["step"]

    return model, optimizer, start_step
