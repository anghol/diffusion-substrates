import os, shutil, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from diffusion import Diffusion


def train(
        denoise_model: torch.nn.Module,
        diffusion: Diffusion,
        timesteps: int,
        train_loader: DataLoader,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        device: str,
        log_path: str,
        log_interval: int
    ):

    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    
    loss_progress = []

    for epoch in range(epochs):
        print(f'----- Epoch {epoch + 1} -----')
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            batch_size = batch.shape[0]
            channels = batch.shape[1]
            image_size = batch.shape[-1]
            batch = batch.to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = diffusion.p_losses(denoise_model, batch, t, loss_type="huber")
            loss_progress.append(loss.item())

            if log_interval:
                if step % log_interval == 0:
                    print("Loss:", loss.item())

            loss.backward()
            optimizer.step()

        logging(log_path, denoise_model, diffusion, epoch, image_size, channels)
    
    return loss_progress


def logging(
        log_path: str, denoise_model: torch.nn.Module, diffusion: Diffusion, epoch: int, image_size: int, channels: int, n_samples=16
    ):

    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    epoch_log_path = log_path + f"/epoch_{epoch}.png"

    samples = diffusion.sample(
        denoise_model, image_size=image_size, batch_size=n_samples, channels=channels
    )
    samples = torch.tensor(samples[-1])
    samples = torch.clamp(samples, -1, 1)
    samples = (samples + 1) / 2

    grid = make_grid(samples, nrow=int(np.sqrt(n_samples)), pad_value=1)
    grid = transforms.ToPILImage()(grid)
    grid.save(epoch_log_path)


def show_grid_samples(n_samples: int, diffusion: Diffusion, model: torch.nn.Module, image_size: int, channels: int, filename='', save_dir=os.getcwd()):
    samples = diffusion.sample(
        model, image_size=image_size, batch_size=n_samples, channels=channels
    )
    samples_T = torch.tensor(samples[-1])
    samples_T = torch.clamp(samples_T, -1, 1)
    samples_T = (samples_T + 1) / 2

    grid = make_grid(samples_T, nrow=int(np.sqrt(n_samples)), pad_value=1)
    grid = transforms.ToPILImage()(grid)
    grid.show()

    if filename:
        path = f'{save_dir}/{filename}.png'
        grid.save(path)

    # rows = cols = int(np.sqrt(n_samples))
    # fig, axes = plt.subplots(rows, cols, figsize=(2.5*rows, 2.5*cols), sharex=True, sharey=True)

    # for ax, img in zip(axes.flat, samples_T):
    #     ax.imshow(img.reshape(image_size, image_size, channels), cmap="gray")
    # plt.show()

    return samples


def generate_and_save_samples(
    samples_dir: str,
    n_samples: int,
    batch_size: int,
    diffusion: Diffusion,
    model: torch.nn.Module,
    image_size: int,
    channels: int,
):
    last_batch = n_samples % batch_size
    count = n_samples // batch_size if last_batch == 0 else n_samples // batch_size + 1
    idx = 0

    for step in range(count):
        if step == count - 1 and last_batch != 0:
            samples = diffusion.sample(
                model, image_size=image_size, batch_size=last_batch, channels=channels
            )
        else:
            samples = diffusion.sample(
                model, image_size=image_size, batch_size=batch_size, channels=channels
            )

        samples_T = torch.tensor(samples[-1])
        samples_T = torch.clamp(samples_T, -1, 1)
        samples_T = (samples_T + 1) / 2

        for sample in samples_T:
            save_image(sample, f"{samples_dir}/sample_{idx}.png")
            idx += 1
    

def create_animation(animation_name: str, samples_all_steps: list, idx: int, timesteps: int):
    image_size = samples_all_steps[0].shape[-1]
    channels = samples_all_steps[0].shape[1]

    fig = plt.figure()
    ims = []
    for t in range(timesteps):
        im = plt.imshow(
            samples_all_steps[t][idx].reshape(image_size, image_size, channels),
            cmap="gray",
            animated=True,
        )
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save(f'{animation_name}.gif')


def count_fid_metric(
        dataloader: DataLoader,
        diffusion: Diffusion,
        model: torch.nn.Module,
        image_size: int,
        channels: int,
        min_n_samples=1000,
        n_feature=2048,
        normalize=False
    ):
    """ Compute the FID metric for evaluation of the diffusion models """
    
    batch_size = next(iter(dataloader)).shape[0]
    count = int(np.ceil(min_n_samples / batch_size))

    real_imgs, fake_imgs = [], []

    start = time.time()
    for i in range(count):
        images = next(iter(dataloader))
        real_imgs.append(images)

        samples = diffusion.sample(model, image_size, batch_size, channels)[-1]
        fake_imgs.append(samples)
    stop = time.time()

    real_imgs = torch.cat(real_imgs)
    real_imgs = (real_imgs + 1) / 2  # to [0, 1]
    real_imgs = real_imgs if normalize else (real_imgs * 255).to(torch.uint8)

    fake_imgs = np.concatenate(fake_imgs)
    fake_imgs = torch.clamp(torch.tensor(fake_imgs), -1, 1)
    fake_imgs = (fake_imgs + 1) / 2  # to [0, 1]
    fake_imgs = fake_imgs if normalize else (fake_imgs * 255).to(torch.uint8)
    print(f"{fake_imgs.shape[0]} samples are generated, time = {stop - start :.3f} sec")

    if channels != 3:
        real_imgs = real_imgs.repeat(1, 3, 1, 1)
        fake_imgs = fake_imgs.repeat(1, 3, 1, 1)

    fid = FrechetInceptionDistance(feature=n_feature, normalize=normalize)
    start = time.time()
    fid.update(real_imgs, real=True)
    fid.update(fake_imgs, real=False)
    metric = fid.compute()
    stop = time.time()
    print(f"FID is computed, time = {stop - start :.3f} sec")

    return metric