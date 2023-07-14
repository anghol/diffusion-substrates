import os, argparse, time
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image


def get_part_substrates_from_one(
    image: torch.Tensor,
    file_idx: int,
    save_dir: str,
    part_size: tuple | list,
    stride: int,
):
    """Cut small partial substrates from one big substrate image."""

    height, width = image.shape[-2], image.shape[-1]
    for i in range((width - (part_size[1] - stride)) // stride):
        for j in range((height - (part_size[0] - stride)) // stride):
            part_image = image[
                :,
                j * stride : j * stride + part_size[0],
                i * stride : i * stride + part_size[1],
            ]

            # Square of the black bacteria's area
            if torch.all(part_image == 0, dim=0).sum() < 100:
                save_image(part_image, f"{save_dir}/part_substrate_{file_idx}.png")
                file_idx += 1
    return file_idx


def get_part_substrates_from_all(
    substrate_dir: str, part_substrate_dir: str, part_size=(256, 256), stride=40
) -> None:
    """Cut small partial substrates from all big substrate images in the directory."""
    if not os.path.isdir(part_substrate_dir):
        os.mkdir(part_substrate_dir)

    idx = 0
    for file in sorted(os.listdir(substrate_dir)):
        with Image.open(f"{substrate_dir}/{file}") as img:
            tensor_image = transforms.ToTensor()(img)
            idx = get_part_substrates_from_one(
                tensor_image, idx, part_substrate_dir, part_size, stride
            )
    print("Part substrates cutted successfully !")

def main():
    parser = argparse.ArgumentParser(description="Get partial substrates from big ones avoiding black bacteria's areas")
    parser.add_argument('source_substrate_dir', type=str, help='Input dir of substrates with removed bacterias')
    parser.add_argument('dest_partsubstrate_dir', type=str, help='Output dir for images of partial substrates')
    parser.add_argument('--part_size', default=(256, 256), type=tuple, help='Size of output partial substrates (in pixels)')
    parser.add_argument('--stride', default=40, type=int, help='Shift for cutting partial substrates (in pixels)')
    args = parser.parse_args()
    
    substrate_dir = args.source_substrate_dir
    part_substrate_dir = args.dest_partsubstrate_dir
    part_size = args.part_size
    stride = args.stride

    start = time.time()
    get_part_substrates_from_all(substrate_dir, part_substrate_dir, part_size, stride)
    print(f'Time = {time.time() - start :.3f} sec')

if __name__ == "__main__":
    main()