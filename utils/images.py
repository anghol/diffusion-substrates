import torch
import numpy as np
from PIL import Image

def image_to_tensor(img: Image.Image) -> torch.Tensor:
    tensor = torch.tensor(np.array(img) / 255)
    return tensor[None, None, :, :]


def tensor_to_image(tensor: torch.Tensor):
    image = tensor.squeeze()
    return Image.fromarray(image.numpy())