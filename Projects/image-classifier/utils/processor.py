
from enum import Enum
from rich.console import Console
from pathlib import Path
from typing import List, Optional, Union
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
        image_path: Union[str, List[str]] = None,
        mode: MODE = Optional[MODE.INFERENCE],
        size: Optional[int] = None,
        resize: Optional[int] = None,
        mean: List[float] = [0.485, 0.485, 0.485],
        std: List[float] = [0.229, 0.224, 0.225],
        batch_size: int = 32,
        shuffle: bool = False
    ) -> DataLoader:
    """
        Preprocess images for training or inference.

        Args:
            image_path: Path to image(s) or list of paths

            mode: 'train' for training preprocessing, 'inference' for inference preprocessing
            
            size: Final image size after cropping (e.g., 224)
            
            resize: Resize shortest side to this value (e.g., 256)
            
            mean: Normalization mean for each channel

            std: Normalization std for each channel

            batch_size: Batch size for DataLoader

            shuffle: Whether to shuffle the data in DataLoader
        
        Returns:
            DataLoader: PyTorch DataLoader with processed images
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
    
    if isinstance(image_path, str):
        base_path = Path(image_path) 
        full_path = base_path / subdir
        if full_path.exists and full_path.is_dir():
            target_path = full_path
        elif base_path.is_dir():
            target_path = base_path
        else:
            raise ValueError("Invalid image_path.")
        console.log(f"[blue]Processing images from directory: {target_path}[/blue]")
        image_datasets = datasets.ImageFolder(target_path, transform=transforms)
    else:
        raise ValueError("image_path must be a string path to directory.")


    if len(image_datasets) == 0:
        raise ValueError(f"No images found in the {target_path}directory.")

    dataloader = DataLoader(
        image_datasets, 
        batch_size, 
        shuffle
    )
    return dataloader
