import os
import numpy as np
import matplotlib.pyplot as plt

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
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for ax, img in zip(axes.flat, samples[-1]):
        ax.imshow(img.reshape(image_size, image_size, channels), cmap="gray")
    plt.show()

    return samples