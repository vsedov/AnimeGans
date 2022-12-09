import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from scipy.stats import entropy
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import utils as vutils
from torchvision.models import Inception_V3_Weights
from torchvision.models.inception import inception_v3
from tqdm import tqdm

import wandb
from src.initial.gans.data.pre_proc import Data  # data class that we have built
from src.initial.gans.models.dcgan import Discriminator, Generator

check_point_path = f"{os.path.dirname(os.path.abspath(__file__))}/checkpoints"
torch.cuda.empty_cache()


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size

    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N//splits):(k+1) * (N//splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def learning_init_setup():
    main_seed = 20220215
    np.random.seed(42)
    torch.manual_seed(main_seed)
    torch.cuda.manual_seed(main_seed)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def load_and_test_model(
        check_point_path, name_space, check_point_num=750, batch_size=None, dataload_amount=10, use_wandb=False):
    # Set up the learning environment
    learning_init_setup()

    # Load the saved model and optimizers
    torch_save = torch.load(f"{check_point_path}/{name_space}/checkpoint_{check_point_num}.pth")
    dis_state = torch_save["dis_state_dict"]
    gen_state = torch_save["gen_state_dict"]

    if use_wandb:
        api = wandb.Api()
        run = api.run("vsedov/dcgan/3tpjvxm0")
        config = run.config
        wandb.init(project="dcgan", name=name_space + "test", config=config)
    else:
        config = {
            'output': 3,
            'ds_type': 'human',
            'test_mode': False,
            'batch_size': batch_size,
            'image_size': 64,
            'epoch_amount': 1000,
            'noise_vector': 64,
            'latent_vector': 124
        }

    image_size = int(config.get("image_size"))
    latent_vector = int(config.get("latent_vector"))
    batch_size = batch_size
    ds_type = config.get("ds_type")
    noise_vector = config.get("noise_vector")

    # Create the data loader
    data = Data()
    data(ds_type=ds_type, view_train=False)
    dataloader = data.get_dl(batch_size, ds_type=ds_type)

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    print(device)
    gen = Generator(latent_vector, image_size, 3).to(device)
    dis = Discriminator(image_size, 3).to(device)

    # Set the state dictionaries of the generator and discriminator
    gen.load_state_dict(gen_state)
    dis.load_state_dict(dis_state)

    # Move the model to the specified device
    gen = gen.to(device)
    dis = dis.to(device)

    # Compare the original and generated images for 100 images
    fake_images = []
    real_images = []
    gen_noise = torch.randn(noise_vector, latent_vector, 1, 1, device=device)
    # for i in tqdm(range(dataload_amount)):
    #     real_batch, _ = next(iter(dataloader))
    # for i, (real_batch, _) in enumerate(dataloader):
    for i, (real_batch, _) in tqdm(enumerate(dataloader)):
        if i == dataload_amount:
            break
        with torch.no_grad():
            fake_batch = gen(gen_noise).detach().cpu()
            fake_images.append(vutils.make_grid(fake_batch, padding=2, normalize=True))
        real_images.append(vutils.make_grid(real_batch, padding=2, normalize=True))

    real_images = [image.cpu() for image in real_images]
    fake_images = [image.cpu() for image in fake_images]

    fake_inception_score = inception_score(fake_images, cuda=True, batch_size=batch_size, resize=True, splits=1)
    real_inception_score = inception_score(real_images, cuda=True, batch_size=batch_size, resize=True, splits=1)

    print(f"Fake Inception Score: {fake_inception_score}")
    print(f"Real Inception Score: {real_inception_score}")

    compare_score = fake_inception_score[0] - real_inception_score[0]
    print(f"Compare Score: {compare_score}")

    output_images(real_images, fake_images, show_real=False)


def output_images(real_images, fake_images, show_real=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    ax1.axis("off")
    ax2.axis("off")

    # Create an animation with the real and fake images
    ims = []
    for i, (real, fake) in enumerate(zip(real_images, fake_images)):
        im2 = ax2.imshow(np.transpose(fake, (1, 2, 0)), animated=True)
        container = [im2]
        if show_real:
            im1 = ax1.imshow(np.transpose(real, (1, 2, 0)), animated=True)
            container.append(im1)
        ims.append(container)

    ani = animation.ArtistAnimation(fig, ims, interval=300, repeat_delay=1000, blit=True)
    plt.show()


def setup(check_point_amount=250, use_wandb=False, batch_size=64, dataload_amount=128):
    load_and_test_model(
        check_point_path,
        "dcgan_human_1000",
        check_point_num=250,
        batch_size=batch_size,
        dataload_amount=dataload_amount,
        use_wandb=use_wandb)
