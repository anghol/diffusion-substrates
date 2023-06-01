import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

import os
from PIL import Image

def get_part_substrates_from_one(image: torch.Tensor, file_idx: int, save_dir: str, part_size=(128, 128), stride=128):
    height, width = image.shape[-2], image.shape[-1]
    for i in range((width - (part_size[1] - stride)) // stride):
        for j in range((height - (part_size[0] - stride)) // stride):
            part_image = image[:, j*stride : j*stride + part_size[0], i*stride : i*stride + part_size[1]]
            save_image(part_image, f'{save_dir}/part_substrate_{file_idx}.png')
            file_idx += 1
    return file_idx

def get_part_substrates_from_all(substrate_dir: str, part_substrate_dir: str, part_size=(128,128), stride=128) -> None:
    idx = 0
    for file in os.listdir(substrate_dir):
        with Image.open(f'{substrate_dir}/{file}') as image:
            tensor_image = transforms.ToTensor()(image)
            idx = get_part_substrates_from_one(tensor_image, idx, part_substrate_dir, part_size, stride)
    print('Part substrates cutted successfully !')


class SubstratesImageDataset(Dataset):
    """ Custom dataset for images with substrates

    Params:
    img_dir - a directory with source images
    transform - transformations which apply to images
    
    Return:
    image - tourch.Tensor    
    """

    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = os.path.join(self.img_dir, f'part_substrate_{idx}.png')
        image = read_image(img_path, mode=ImageReadMode.GRAY)
        # image = read_image(img_path)   
        if self.transform:
            image = self.transform(image)
        return image

def create_substrate_dataset(part_substrate_dir: str):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    dataset = SubstratesImageDataset(part_substrate_dir, transform)
    return dataset