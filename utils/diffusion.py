import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance


def logging(log_path, diffusion, model, epoch, n_samples, image_size, channels):
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    epoch_log_path = log_path + f"/epoch_{epoch}"
    os.mkdir(epoch_log_path)

    samples = diffusion.sample(
        model, image_size=image_size, batch_size=n_samples, channels=channels
    )
    for idx, img in enumerate(samples[-1]):
        plt.imsave(
            epoch_log_path + f"/sample_{idx}.png",
            img.reshape(image_size, image_size),
            cmap="gray",
        )


def show_samples(n_samples, diffusion, model, image_size, channels):
    samples = diffusion.sample(
        model, image_size=image_size, batch_size=n_samples, channels=channels
    )
    rows = cols = int(np.sqrt(n_samples))
    fig, axes = plt.subplots(rows, cols, figsize=(2.5*rows, 2.5*cols))

    for ax, img in zip(axes.flat, samples[-1]):
        ax.imshow(img.reshape(image_size, image_size, channels), cmap="gray")
    plt.show()

    return samples


def create_animation(animation_name, samples_all_steps, idx, timesteps):
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


def count_fid_metric(dataloader, diffusion, model, image_size, channels, n_feature):
    """ Compute the FID metric for evaluation of the diffusion models """
    
    real_img = next(iter(dataloader))
    n_samples, channels = real_img.shape[0], real_img.shape[1]
    real_img = (real_img + 1) / 2 * 255
    if channels != 3:
        real_img = real_img.byte().repeat(1, 3, 1, 1)
    
    samples = diffusion.sample(model, image_size=image_size, batch_size=n_samples, channels=channels)[-1]
    samples = torch.clamp(torch.tensor(samples), -1, 1)
    samples = (samples + 1) / 2 * 255
    if channels != 3:
        samples = samples.byte().repeat(1, 3, 1, 1)

    if isinstance(n_feature, int):
        n_feature = [n_feature]

    metric = []
    for feature in n_feature:
        fid = FrechetInceptionDistance(feature=feature)
        fid.update(real_img, real=True)
        fid.update(samples, real=False)
        metric.append(fid.compute())
    return metric
