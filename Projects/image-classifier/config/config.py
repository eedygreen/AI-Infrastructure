import os
import torch
from typing import Optional
from enum import Enum
from torch import optim, nn
from middleware import Classifier
from torchvision import datasets, transforms, models

class LossFunction(str, Enum):
    NLL = "nll"
    CROSS_ENTROPY = "ce"
    MSE = "mse"
    BCE = "bce"

def create_criterion(loss_type: LossFunction) -> nn.Module:
    """
        Create a Loss Function from Enum choice
    """
    loss_function = {
        LossFunction.NLL: nn.NLLLoss(),
        LossFunction.CROSS_ENTROPY: nn.CrossEntropyLoss(),
        LossFunction.MSE: nn.MSELoss(),
        LossFunction.BCE: nn.BCELoss(),
    }
    return loss_function[loss_type]

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.224, 0.225])
])

def get_data_loaders(
        data_dir: str, 
        batch_size: int, 
        shuffle: bool
    ) -> tuple:
    """ 
    Get data loaders for training, validation, and testing datasets.

    Arguements:
    1. batch_size: int - The batch size for the data loaders.

    Returns:
        A tuple containing DataLoaders for training, validation, and testing datasets.
    """

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)

    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=batch_size, shuffle=shuffle)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size, shuffle=shuffle)
    valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=batch_size, shuffle=shuffle)

    class_to_idx = train_image_datasets.class_to_idx #
    
    return train_dataloaders, valid_dataloaders, test_dataloaders, class_to_idx

def get_model(
        model_name: str,
        weights: str,
        pretrained: bool
    ) -> models:
    """
    Get a pre-trained model from torchvision.models

    Arguements:
    1. model: str - The model architecture to use (e.g., 'vgg16', 'densenet121').
    2. weights: str - The weights to load for the model (e.g., 'VGG16_Weights.IMAGENET1K_V1').
    
    Returns:
        A pre-trained model with the specified architecture and weights.
    """
    model_class = getattr(models, model_name)

    if weights and isinstance(weights, str):
        weights_class = getattr(models, weights.split('.')[0])
        weights = getattr(weights_class, weights.split('.')[1])
        
    return model_class(weights=weights if pretrained else None)

def setup_model_data(
        data_dir: str,
        arch: str,
        weights: str,
        learning_rate: float,
        input_size: int,
        output_size: int,
        hidden_size: int,
        min_hidden_size: int,
        pretrained: Optional[bool] = False,
        use_batch_norm: Optional[bool] = False,
        dropout: Optional[float] = 0.5,
        gpu: Optional[bool] = False,
        shuffle: Optional[bool] = True,
        criterion: Optional[LossFunction] = LossFunction.NLL,
        batch_size: Optional[int] = 32
    ) -> tuple:
    """
        Setup model and data loaders based on current parameters.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    train_dataloaders, valid_dataloaders, test_dataloaders, class_to_idx = get_data_loaders(data_dir=data_dir, batch_size=batch_size, shuffle=shuffle)

    model = get_model(model_name=arch, weights=weights, pretrained=pretrained)

    """ Freeze parameters so we don't backprop through them """
    for param in model.parameters():
        param.requires_grad = False

    if input_size is None or input_size <= 0:
        input_size = model.classifier[0].in_features

    if output_size is None:
        output_size = len(os.listdir(os.path.join(data_dir, 'train')))

    classifier = Classifier(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        min_hidden_size=min_hidden_size,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
    )

    model.classifier = classifier
    
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)

    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    model.to(device)

    # results = {
    #     'model': model, 
    #     'train_dataloaders': train_dataloaders, 
    #     'valid_dataloaders': valid_dataloaders, 
    #     'test_dataloaders': test_dataloaders, 
    #     'criterion': criterion, 
    #     'optimizer': optimizer, 
    #     'device': device
    # }
    return model, train_dataloaders, valid_dataloaders, test_dataloaders, criterion, optimizer, device, class_to_idx
