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


def get_random_label(batch_size, hair_classes, eye_classes):
    """Sample a batch of random class labels for hair and eye color.

    Args:
        batch_size (int): number of labels to sample.
        hair_classes (int): number of hair color classes.
        eye_classes (int): number of eye color classes.

    Returns:
        torch.Tensor: A tensor of size (batch_size, hair_classes + eye_classes) representing the one-hot encoded class labels.
    """
    hair_code = torch.zeros(batch_size, hair_classes)
    eye_code = torch.zeros(batch_size, eye_classes)

    hair_type = torch.randint(0, hair_classes, (batch_size,))
    eye_type = torch.randint(0, eye_classes, (batch_size,))

    hair_code[range(batch_size), hair_type] = 1
    eye_code[range(batch_size), eye_type] = 1

    return torch.cat((hair_code, eye_code), dim=1)


def generate_by_attributes(
    model,
    device,
    latent_dim,
    hair_classes,
    eye_classes,
    sample_dir,
    step=None,
    hair_color=None,
    eye_color=None,
):
    """
    Generates image samples with fixed attributes.

    Parameters:
        model (nn.Module): The model to generate images.
        device (torch.device): The device to run the model on.
        latent_dim (int): The dimension of the noise vector.
        hair_classes (int): The number of hair colors.
        eye_classes (int): The number of eye colors.
        sample_dir (str): The folder to save images.
        step (int, optional): The current training step. Defaults to None.
        hair_color (str, optional): The chosen hair color. If None, hair color will be randomly selected.
        eye_color (str, optional): The chosen eye color. If None, eye color will be randomly selected.

    Returns:
        None
    """
    # Choose hair and eye class
    hair_class = hair_dict.get(hair_color, np.random.choice(hair_classes))
    eye_class = eye_dict.get(eye_color, np.random.choice(eye_classes))

    # Generate hair and eye tags
    hair_tag = torch.zeros(64, hair_classes).to(device)
    eye_tag = torch.zeros(64, eye_classes).to(device)
    for i in range(64):
        hair_tag[i, hair_class] = 1
        eye_tag[i, eye_class] = 1

    # Concatenate hair and eye tags
    tag = torch.cat((hair_tag, eye_tag), dim=1)
    z = torch.randn(64, latent_dim).to(device)

    # Generate image
    output = model(z, tag)
    filename = f"{sample_dir}/{hair_mapping[hair_class]} hair {eye_mapping[eye_class]} eyes.png"
    vutils.save_image(output, filename)


def hair_grad(model, device, latent_dim, hair_classes, eye_classes, sample_dir):
    """Generate image samples with fixed eye class and noise, change hair color.

    Args:
        model (nn.Module): model to generate images.
        device (torch.device): device to run model on.
        latent_dim (int): dimension of the noise vector.
        hair_classes (int): number of hair colors.
        eye_classes (int): number of eye colors.
        sample_dir (str): folder to save images.

    Returns:
        None
    """
    eye = torch.zeros(eye_classes, dtype=torch.float32).to(device)
    eye[np.random.randint(eye_classes)] = 1
    eye = eye.unsqueeze(0)

    z = torch.randn(latent_dim, dtype=torch.float32).unsqueeze(0).to(device)
    img_list = []
    for i in range(hair_classes):
        hair = torch.zeros(hair_classes, dtype=torch.float32).to(device)
        hair[i] = 1
        hair = hair.unsqueeze(0)
        tag = torch.cat((hair, eye), 1)
        img_list.append(model(z, tag))

    output = torch.cat(img_list, dim=0)
    vutils.save_image(
        output, f"{sample_dir}/change_hair_color.png", nrow=hair_classes
    )


def eye_grad(model, device, latent_dim, hair_classes, eye_classes, sample_dir):
    """Generate random image samples with fixed hair class and noise, change eye color.

    Args:
        model (nn.Module): model to generate images.
        device (torch.device): device to run model on.
        latent_dim (int): dimension of the noise vector.
        hair_classes (int): number of hair colors.
        eye_classes (int): number of eye colors.
        sample_dir (str): folder to save images.

    Returns:
        None
    """
    hair = torch.zeros(hair_classes, dtype=torch.float32).to(device)
    hair[np.random.randint(hair_classes)] = 1
    hair = hair.unsqueeze(0)

    z = torch.randn(latent_dim, dtype=torch.float32).unsqueeze(0).to(device)
    img_list = []
    for i in range(eye_classes):
        eye = torch.zeros(eye_classes, dtype=torch.float32).to(device)
        eye[i] = 1
        eye = eye.unsqueeze(0)
        tag = torch.cat((hair, eye), 1)
        img_list.append(model(z, tag))

    output = torch.cat(img_list, dim=0)
    vutils.save_image(
        output, f"{sample_dir}/change_eye_color.png", nrow=eye_classes
    )


def fix_noise(model, device, latent_dim, hair_classes, eye_classes, sample_dir):
    """Generate random image samples with fixed noise.
    Args:
        model (nn.Module): model to generate images.
        device (torch.device): device to run model on.
        latent_dim (int): dimension of the noise vector.
        hair_classes (int): number of hair colors.
        eye_classes (int): number of eye colors.
        sample_dir (str): folder to save images.

    Returns:
        None
    """

    z = (
        torch.randn(latent_dim)
        .unsqueeze(0)
        .to(device)
        .type(torch.cuda.FloatTensor)
    )

    print(type(z))

    img_list = []
    for i in range(eye_classes):
        for j in range(hair_classes):
            eye = torch.zeros(eye_classes).to(device)
            hair = torch.zeros(hair_classes).to(device)
            eye[i], hair[j] = 1, 1
            eye.unsqueeze_(0)
            hair.unsqueeze_(0)

            tag = torch.cat((hair, eye), 1)
            img_list.append(model(z, tag))

    print(len(img_list))

    output = torch.cat(img_list, dim=0)
    vutils.save_image(output, f"{sample_dir}/fix_noise.png", nrow=hair_classes)


def interpolate(
    model, device, latent_dim, hair_classes, eye_classes, sample_dir, samples=10
):
    """Interpolate between two random latent vectors and generate image samples.

    Args:
        model (nn.Module): model to generate images.
        device (torch.device): device to run model on.
        latent_dim (int): dimension of the noise vector.
        hair_classes (int): number of hair colors.
        eye_classes (int): number of eye colors.
        sample_dir (str): folder to save images.
        samples (int, optional): number of image samples to generate. Defaults to 10.

    Returns:
        None
    """

    def generate_random_vector(latent_dim, hair_classes, eye_classes, device):
        """Generate a random latent vector and a random class vector for hair and eye color.

        Args:
            latent_dim (int): dimension of the noise vector.
            hair_classes (int): number of hair colors.
            eye_classes (int): number of eye colors.
            device (torch.device): device to run model on.

        Returns:
            tuple: tuple containing the random latent vector and class vector.
        """
        z = torch.randn(1, latent_dim).to(device)
        h = torch.zeros(1, hair_classes).to(device)
        e = torch.zeros(1, eye_classes).to(device)
        h[0][np.random.randint(hair_classes)] = 1
        e[0][np.random.randint(eye_classes)] = 1
        c = torch.cat((h, e), 1)
        return z, c

    z1, c1 = generate_random_vector(
        latent_dim, hair_classes, eye_classes, device
    )
    z2, _ = generate_random_vector(
        latent_dim, hair_classes, eye_classes, device
    )
