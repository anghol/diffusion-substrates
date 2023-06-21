import os
import torch
import numpy as np
from PIL import Image

def image_to_tensor(img: Image.Image) -> torch.Tensor:
    tensor = torch.tensor(np.array(img) / 255)
    return tensor[None, None, :, :]


def tensor_to_image(tensor: torch.Tensor):
    image = tensor.squeeze()
    return Image.fromarray(image.numpy())


def rename_images(suffix: str, path='data/separated_cells'):
    """ Rename all images in necessary directory.\n
    Params:
    suffix - The suffix of filename e.g. cell_0.png where 'cell' is suffix.
    path - The path to the directory with images.
    """

    filename = os.listdir(path)[0]
    img_format = filename.split('.')[-1]

    for i, filename in enumerate(sorted(os.listdir(path))):
        old_name = os.path.join(path, filename)
        new_name = os.path.join(path, suffix + f'_{i}.{img_format}')
        os.rename(old_name, new_name)