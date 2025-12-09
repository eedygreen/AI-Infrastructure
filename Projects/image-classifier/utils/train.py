from typing import Optional
from rich.console import Console
from config import setup_model_data, LossFunction
from utils import run_validation

console = Console()

def train_model(
        data_dir: str,
        arch: str,
        weights: str,
        learning_rate: float,
        epochs: int,
        print_every: int,
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
        batch_size: Optional[int] = 32,
        save_dir: Optional[str] = None
):
    """
    Train the neural network model.

    Usage:\n
        python -m cli train --epochs 5 --learning-rate 0.001 \n
        python -m cli train --arch <architecture> --epochs <epochs>  --criterion <criterion> --print_every <print_every>
 
    """
    console.log("Setting up model and data loaders ...")

    model, dataloaders, _, test_dataloaders, criterion, optimizer, device = setup_model_data(
        data_dir=data_dir,
        arch=arch,
        weights=weights,
        learning_rate=learning_rate,
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        min_hidden_size=min_hidden_size,
        pretrained=pretrained,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        gpu=gpu,
        shuffle=shuffle,
        criterion=criterion,
        batch_size=batch_size
    )

    console.log(f"Starting training on device: {device}")
    console.log(f"Training for {epochs} epochs with learning rate {learning_rate}")

    running_loss = 0
    step = 0

    for epoch in range(epochs):
        
        model.train()
        
        for images, labels in dataloaders:
            step += 1
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    
            optimizer.zero_grad()
            model_prediction = model(images)
            loss = criterion(model_prediction, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if step % print_every == 0:
                accuracy, test_loss = run_validation(model, test_dataloaders, criterion, device)
     
                console.log(
                    f"Epoch: {epoch + 1}/{epochs} \
                    Step: {step}\
                    Training_Loss: {running_loss / print_every:.3f}\
                    Test_Loss: {test_loss / len(test_dataloaders):.3f}\
                    Accuracy: {accuracy / len(test_dataloaders):.2f}"
                )
                running_loss = 0
                model.train()

    console.log("Training completed.")
    if save_dir:
        console.log("Saving checkpoint...")
        model.classifier.save_checkpoint(save_dir=save_dir, epoch=epochs, optimizer=optimizer)