import typer
from rich.console import Console
from utils.model_setup import setup_model_data
from utils.validation import run_validation

console = Console()

def train(
    ctx: typer.Context,
    epochs: int,
    learning_rate: float,
    print_every: int,
    model,
    dataloaders,
    test_dataloaders,
    criterion,
    optimizer,
    device
):
    """
    Train the neural network model.

    Usage:\n
        python -m cli train --epochs 5 --learning-rate 0.001 \n
        python -m cli train --arch <architecture> --epochs <epochs>  --criterion <criterion> --print_every <print_every>
 
    """
    epochs = ctx.obj["epochs"]
    print_every = ctx.obj["print_every"]

    console.log("Setting up model and data loaders ...")

    model, dataloaders, _, test_dataloaders, criterion, optimizer, device, _ = setup_model_data(ctx)

    console.log(f"Starting training on device: {device}")
    console.log(f"Training for {epochs} epochs with learning rate {ctx.obj["learning_rate"]}")

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