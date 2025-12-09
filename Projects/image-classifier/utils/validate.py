from config import setup_model_data
import torch
from rich.console import Console

console = Console()

def run_validation(
        model, 
        test_dataloaders, 
        criterion, 
        device
    )-> tuple:
    """
    Validate the model on the test dataset.

    Arguements:
        model: The neural network model to validate.
        test_dataloaders: DataLoader for the test dataset.
        criterion: Loss function used for validation.
        device: The device to run the validation on (CPU or GPU).
    """

    test_loss = 0
    accuracy = 0
    
    console.log("Starting validation ...")

    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloaders:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                        
            prediction = model(images)
            test_loss += criterion(prediction, labels).item()
                        
            ps = torch.exp(prediction)
            _, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

    return accuracy, test_loss