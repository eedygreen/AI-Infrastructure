
from enum import Enum
import os
from rich.console import Console
from pathlib import Path
from typing import List, Optional, Dict
from torch.utils.data import DataLoader
from torchvision import datasets
from processor import(
    Compose, 
    RandomRotation, 
    RandomResizedCrop,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
    Resize,
    CenterCrop
)

console = Console()

class MODE(str, Enum):
    TRAINING = "train"
    INFERENCE = "inference"

def process_image(
        image_path: str = None,
        mode: MODE = Optional[MODE.INFERENCE],
        size: Optional[int] = 256,
        resize: Optional[int] = 224,
        mean: List[float] = [0.485, 0.485, 0.485],
        std: List[float] = [0.229, 0.224, 0.225],
        batch_size: int = 32,
        shuffle: bool = False
    ) -> Dict:
    """
        Preprocess images for training or inference.

        Args:
            image_path: Directory containing images.

            mode: 'train' for training preprocessing, 'inference' for inference preprocessing
            
            size: Final image size after cropping (e.g., 224)
            
            resize: Resize shortest side to this value (e.g., 256)
            
            mean: Normalization mean for each channel

            std: Normalization std for each channel

            batch_size: Batch size for DataLoader

            shuffle: Whether to shuffle the data in DataLoader
        
        Returns:
            DataLoader: PyTorch DataLoader with processed images
            class_to_idx: mapping of classes to indices
    """
    if mode == MODE.TRAINING:
        transforms = Compose([
            RandomRotation(30),
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        subdir = 'train'
    elif mode == MODE.INFERENCE:
        resize = resize or (size + 32)
        transforms = Compose([
            Resize(resize),
            CenterCrop(size),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        subdir = 'test'
    else:
        raise ValueError("Invalid mode. Choose either 'train' or 'inference'.")

    image_path = Path(image_path)
    if image_path.is_dir():
        image_path = os.path.join(image_path, subdir)

    console.log(f"[blue]Processing images from directory: {image_path}[/blue]")

    image_datasets = datasets.ImageFolder(image_path, transform=transforms)

    if len(image_datasets) == 0:
        raise ValueError(f"No images found in the {image_path} directory.")

    class_to_idx = image_datasets.class_to_idx

    dataloader = DataLoader(
        image_datasets, 
        batch_size, 
        shuffle,
    )
    return dataloader, class_to_idx
