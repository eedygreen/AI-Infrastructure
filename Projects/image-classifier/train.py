'''
1. Train
Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains

Options: * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory 
* Choose architecture: python train.py data_dir --arch "vgg13" 
* Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 
* Use GPU for training: python train.py data_dir --gpu
'''

import torch
import json
from torch import nn, optim
from rich.console import Console
from utils.helper import save_checkpoint
from middleware.network import Classifier
from config.config import get_data_loaders, get_model, ParamtersOptions

console = Console(stderr=True)

param_opts = ParamtersOptions()

dataloaders, valid_dataloaders, test_dataloaders = get_data_loaders()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model = get_model()

""" Freeze parameters so we don't backprop through them """
for param in model.parameters():
    param.requires_grad = False

classifier = Classifier()
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.classifier.parameters(), lr=param_opts.learning_rate)



device = torch.device(param_opts.gpu if torch.cuda.is_available() else "cpu")
model.to(device)

learning_rate: float = param_opts.learning_rate

    # Define transforms for the training, validation, and testing

def validation(model, testdataloaders, criterion) -> tuple:
    """ 
    Validate the model on the test dataset.

    Arguements:
    1. model: The neural network model to validate.
    2. testdataloaders: DataLoader for the test dataset.
    3. criterion: Loss function used for validation.

    Returns:
        accuracy: The accuracy of the model on the test dataset.
        test_loss: The total loss on the test dataset.
    """
    test_loss = 0
    accuracy = 0

    for images, labels in testdataloaders:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    
        prediction = model(images)
        test_loss += criterion(prediction, labels).item()
                    
        ps = torch.exp(prediction)
        _, top_class = ps.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()  

    return accuracy, test_loss


def train(model, 
          train_dataloaders, 
          test_dataloaders, 
          optimizer, criterion, 
          epochs: int=3, 
          print_every: int=5
        ):
    """
    Train the neural network model.

    Usage: train(model, train_dataloaders, test_dataloaders, optimizer, criterion, epochs=3, print_every=5)
    
    Arguements:
    1. model: The neural network model to train.
    2. train_dataloaders: DataLoader for the training dataset.
    3. test_dataloaders: DataLoader for the test dataset.
    4. optimizer: The optimizer used for training.
    5. criterion: Loss function used for training.
    6. epochs: Number of epochs to train the model.
    7. print_every: Frequency of printing training progress.
    """
    running_loss = 0
    step = 0

    for epoch in range(epochs):
        
        model.train()
        
        for images, labels in train_dataloaders:
            step += 1
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    
            optimizer.zero_grad()
            model_prediction = model(images)
            loss = criterion(model_prediction, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if step % print_every == 0:
                model.eval()
                with torch.no_grad():
                    accuracy, test_loss = validation(model, test_dataloaders, criterion)
     
                print(f"Epoch: {epoch + 1}/{epochs} \
                Step: {step}\
                Training_Loss: {running_loss / print_every:.3f}\
                Test_Loss: {test_loss / len(test_dataloaders):.3f}\
                Accuracy: {accuracy / len(test_dataloaders):.2f}\
                ")
                running_loss = 0
                model.train()
    console.log("Training completed.")
    console.log("Saving checkpoint...")
    save_checkpoint()
    console.log("Checkpoint saved.")

