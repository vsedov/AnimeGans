import os
import pickle

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn, optim
from torchvision import utils as vutils

from src.core import hc, hp

#  TODO: (vsedov) (16:31:15 - 24/11/22): Change the names of this dataclass
from src.initial.gans.data.pre_proc import Data  # data class that we have built
from src.initial.gans.models.dcgan import Discriminator, Generator


def learning_init_setup():
    main_seed = 20220215
    np.random.seed(42)
    torch.manual_seed(main_seed)
    torch.cuda.manual_seed(main_seed)
    """Use deterministic Convolutional Algorithms"""
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(
        mode=True,
    )


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    learning_init_setup()

    check_point_path = f"{os.path.dirname(os.path.abspath(__file__))}/checkpoints"

    data = Data()
    data.create_data()
    params = hc.initial["params"]
    device = hc.DEFAULT_DEVICE
    epoch_amount = [2, 4, 100, 300, 500, 1000][4]

    dis, gen = hp.to_default_device(
        Discriminator(params["discriminator"], params["output"]),
        Generator(params["latent_vector"], params["generator"], params["output"]),
    )
    print(gen.device)
    print(dis.device)

    dis.apply(weights_init)
    gen.apply(weights_init)

    fixed_noise = torch.randn(64, params["latent_vector"], 1, 1, device=device)

    # Create mini batch data
    batch_size = [32, 64, 128][0]
    dataloader = data.get_dl(batch_size)
    learning_rate = 0.0001
    criterion = nn.BCELoss()  # Binary cross entropy Loss function

    #  TODO: (vsedov) (19:38:28 - 24/11/22): Refactor this into a more subtle method

    optimizer_d = optim.Adam(dis.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    d_container = train(
        dataloader,
        dis,
        gen,
        criterion,
        optimizer_d,
        optimizer_g,
        epoch_amount,
        fixed_noise,
        device,
        batch_size,
        params,
        check_point_path,
    )

    # Visualise the data
    visualise_training_losss(d_container)
    visualise(d_container)
    pickle_dumb(d_container, check_point_path, "data_container")


def pickle_dumb(data, path, name):
    with open(f"{path}/{name}.pkl", "wb") as f:
        pickle.dump(data, f)


def visualise_training_losss(data_container):
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    axes.plot(data_container["d_loss"], label="Discriminator loss")
    axes.plot(data_container["g_loss"], label="Generator loss")
    axes.set_xlabel("Epochs")
    axes.set_ylabel("Loss")
    axes.legend()
    plt.show()


def visualise(d_container):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in d_container["img_list"]]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save("dcgan.mp4")
    plt.show()


# Might be a base line for my core stuf , im not sure
def train(
    dataloader,
    dis,
    gen,
    criterion,
    optimizer_d,
    optimizer_g,
    epoch_amount,
    fixed_noise,
    device,
    batch_size,
    params,
    check_point_path,
):
    data_container = {"img_list": [], "g_loss": [], "d_loss": [], "iters": 0}
    for epoch in tqdm.tqdm(range(epoch_amount)):
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            dis.train()
            gen.train()
            X, _ = batch
            X = X.to(device)

            dis.zero_grad()
            label = torch.full((X.shape[0],), 1.0, device=device)
            output = dis(X).view(-1)
            error_d_real = criterion(output, label)
            error_d_real.backward()
            derivative_d_real = output.mean().item()
            #  ──────────────────────────────────────────────────────────────────────
            #
            # Generator time
            noise = torch.randn(batch_size, params["latent_vector"], 1, 1, device=device)
            fake = gen(noise)
            label.fill_(0.0)
            output = dis(fake.detach()).view(-1)
            error_d_fake = criterion(output, label)
            error_d_fake.backward()
            derivative_d_fake = output.mean().item()
            error_d = error_d_real + error_d_fake
            optimizer_d.step()

            #  ──────────────────────────────────────────────────────────────────────
            # Generator update time
            gen.zero_grad()
            label.fill_(1)
            output = dis(fake).view(-1)
            error_g = criterion(output, label)
            error_g.backward()
            derivative_g = output.mean().item()
            optimizer_g.step()

            if i % 10 == 0:
                print(
                    f"[{epoch}/{epoch_amount}][{i}/{len(dataloader)}]\t"
                    f"Loss_D: {error_d.item():.4f}\t"
                    f"Loss_G: {error_g.item():.4f}\t"
                    f"D(x): {derivative_d_real:.4f}\t"
                    f"D(G(z)): {derivative_d_fake:.4f} / {derivative_g:.4f}"
                )
                print(tqdm.tqdm._instances)

            #  ──────────────────────────────────────────────────────────────────────
            # Checkpoints are always nice to have
            if i % 200 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": gen.state_dict(),
                        "optimizer_state_dict": optimizer_g.state_dict(),
                        "loss": error_g,
                    },
                    os.path.join(check_point_path, f"generator_{epoch}_{i}.pth"),
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": dis.state_dict(),
                        "optimizer_state_dict": optimizer_d.state_dict(),
                        "loss": error_d,
                    },
                    os.path.join(check_point_path, f"discriminator_{epoch}_{i}.pth"),
                )

            data_container["g_loss"].append(error_g.item())
            data_container["d_loss"].append(error_d.item())
            data_container["iters"] += 1

            dis.eval()
            gen.eval()

        with torch.no_grad():
            fake = gen(fixed_noise).detach().cpu()
        data_container["img_list"].append(vutils.make_grid(fake, padding=2, normalize=True))
    return data_container


if __name__ == "__main__":
    main()
