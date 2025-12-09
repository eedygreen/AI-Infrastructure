
import os, typer
from pathlib import Path
from utils import Checkpoint, process_image, run_validation
from rich.console import Console
from middleware import Classifier
from typing_extensions import Annotated
from typing import Optional, List
from config import setup_model_data, LossFunction, create_criterion

app = typer.Typer(no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]})
console = Console(stderr=True)

@app.callback()
def main(ctx: typer.Context,
    data_dir: Annotated[str, 
        typer.Option(
            "--data-dir", 
            "-d", metavar="", 
            help="Path to the data directory"
        )
    ] = "flowers",
    save_dir: Annotated[str, 
        typer.Option(
            "--save-dir", "-s", 
            metavar="", 
            help="Directory to save saves",
            exists=True,
            dir_okay=True
        )
    ] = "saves",
    file_path: Annotated[Optional[Path], 
        typer.Option(
            "--file-path", "-f",
            metavar="", 
            help="Path to the checkpoint file",
            exists= True,
            dir_okay=False
        )
    ] = None,
    input_size: Annotated[int, 
        typer.Option(
            "--input-size", 
            "-i", 
            metavar="", 
            help="Number of Input Neurons of the classifier"
        )
    ] = 0,
    output_size: Annotated[int, 
        typer.Option(
            "--data-size", 
            "-ds", 
            metavar="", 
            help="Number of sample data"
        )
    ] = None,
    arch: Annotated[str, 
        typer.Option(
            "--arch", 
            "-a", metavar="", 
            help="Model architecture: vgg16, vgg13, densenet121"
        )
    ] = "vgg16",
    weights: Annotated[str, 
        typer.Option(
            "--weights", 
            "-w", metavar="", 
            help="Pre-trained weights to use"
        )
    ] = "VGG16_Weights.IMAGENET1K_V1",
    learning_rate: Annotated[float, 
        typer.Option(
            "--learning-rate", 
            "-lr", metavar="", 
            help="Learning rate for training"
        )
    ] = 0.001,
    dropout: Annotated[float, 
        typer.Option(
            "--dropout", 
            "-dr", 
            metavar="", 
            help="Dropout rate for the classifier"
        )
    ] = 0.5,
    hidden_size: Annotated[int, 
        typer.Option(
            "--hidden-size", 
            "-hu", 
            metavar="", 
            help="Number of hidden neurons in the classifier"
        )
    ] = 512,
    min_hidden_size: Annotated[int,
        typer.Option(
            "--min-size",
            "-ms",
            metavar="",
            help="Minimum Number of hidden neurons in the classifier"
        )
    ] = 64,
    epochs: Annotated[int, 
        typer.Option(
            "--epochs", 
            "-e", 
            metavar="", 
            help="Number of epochs for training"
        )
    ] = 5,
    batch_size: Annotated[int, 
        typer.Option(
            "--batch-size", 
            "-b", 
            metavar="", 
            help="Batch size for data loaders"
        )
    ] = 32,
    use_batch_norm: Annotated[bool,
        typer.Option(
            "--batch-norm", 
            "-bn",
            metavar="",
            help="Enable Batch Normalization. Default is False"
        )
    ] = False,
    criterion: Annotated[LossFunction,
        typer.Option(
            "--criterion",
            "-lf",
            metavar="",
            help="Loss Function to use, Default is nn.NLLLOS()"
        )
    ] = LossFunction.NLL,
    pretrained: Annotated[bool,
        typer.Option(
            "--pretrained_model",
            "-pr-model",
            metavar="",
            help="Enable Pretrained Model"
        )
    ] = True,
    shuffle: Annotated[bool, 
        typer.Option(
            "--shuffle", 
            "-sh", 
            help="Shuffle data in data loaders"
        )
    ] = True,
    print_every: Annotated[int, 
        typer.Option(
            "--print-every", 
            "-p", metavar="", 
            help="Print training info every n steps"
        )
    ] = 5,
    gpu: Annotated[bool,
        typer.Option(
            "--gpu", 
            "-gp",
            metavar="",
            help="Enable GPU Device for Faster training. Default is False (Disable)"
        )
    ] = False
):  
    """
        Image Classifier application.

        Example:\n
            python cli.py train --arch vgg16 --batch-size 64 --learn-rate 0.001 --data-dir data/data \n
            python cli.py validate --arch vgg16 --batch-size 64 --data-dir data/data \n
            python cli.py predict --arch vgg16 --model-path models/best.pth
        """
    """Image Classifier Application"""
    ctx.obj = {
        "data_dir": data_dir,
        "save_dir": save_dir,
        "file_path": file_path,
        "input_size": input_size,
        "output_size": output_size, 
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "min_hidden_size": min_hidden_size,
        "arch": arch,
        "weights": weights,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "epochs": epochs,
        "use_batch_norm": use_batch_norm,
        "criterion": create_criterion(criterion),
        "shuffle": shuffle,
        "print_every": print_every,
        "gpu": gpu,
        "pretrained": pretrained
    }

@app.command()
def validate(
    ctx: typer.Context,
    data_dir: Annotated[str, 
        typer.Option(
            "--data-dir", 
            "-d", 
            metavar="", 
            help="Path to data directory"
        )  
    ] = "flowers",
    criterion: Annotated[LossFunction,
        typer.Option(
            "--criterion",
            "-lf",
            metavar="",
            help="Loss Function to use, Default is nn.NLLLOS()"
        )  ] = LossFunction.NLL,
    input_size: Annotated[int, 
        typer.Option(
            "--input-size", 
            "-i", 
            metavar="", 
            help="Input Neuron size of the classifier"
        )
    ] = 0,
    output_size: Annotated[int, 
        typer.Option(
            "--data-size", 
            "-ds", 
            metavar="", 
            help="Number of sample data"
        )
    ] = None,
    hidden_size: Annotated[int, 
        typer.Option(
            "--hidden-size", 
            "-hu", 
            metavar="", 
            help="Number of hidden neurons in the classifier"
        )
    ] = 512,
    min_hidden_size: Annotated[int,
        typer.Option(
            "--min-size",
            "-ms",
            metavar="",
            help="Minimum Number of hidden neurons in the classifier"
        )
    ] = 64,
    use_batch_norm: Annotated[bool,
        typer.Option(
            "--batch-norm", 
            "-bn",
            metavar="",
            help="Enable Batch Normalization. Default is False"
        )
    ] = False,
    dropout: Annotated[float, 
        typer.Option(   
            "--dropout", 
            "-dr", 
            metavar="", 
            help="Dropout rate for the classifier"
        )
    ] = 0.5,
    gpu: Annotated[bool,
        typer.Option(
            "--gpu", 
            "-gp",
            metavar="",
            help="Enable GPU Device for Faster training. Default is False (Disable)"
        )
    ] = False,
    weights: Annotated[str, 
        typer.Option(
            "--weights", 
            "-w", metavar="", 
            help="Pre-trained weights to use"
        )
    ] = "VGG16_Weights.IMAGENET1K_V1",
    arch: Annotated[str, 
        typer.Option(
            "--arch", 
            "-a", metavar="", 
            help="Model architecture: vgg16, vgg13, densenet121"
        )
    ] = "vgg16",
    shuffle: Annotated[bool, 
        typer.Option(
            "--shuffle", 
            "-sh", 
            help="Shuffle data in data loaders"
        )
    ] = True,
    learning_rate: Annotated[float, 
        typer.Option(
            "--learning-rate", 
            "-lr", metavar="", 
            help="Learning rate for training"
        )
    ] = 0.001,
    pretrained: Annotated[bool,
        typer.Option(
            "--pretrained_model",
            "-pr-model",
            metavar="",
            help="Enable Pretrained Model"
        )
    ] = True
):
    """
    Validate the model on the test dataset.

    Usage:\n
        python -m cli validate \n
        python -m cli validate --arch <architecture> --batch-size <batch_size> --data-dir <data_dir>
    """
    ctx.ensure_object(dict) 
    ctx.obj["arch"] = arch
    ctx.obj["weights"] = weights
    ctx.obj["pretrained"] = pretrained
    ctx.obj["learning_rate"] = learning_rate
    ctx.obj["shuffle"] = shuffle
    ctx.obj["dropout"] = dropout
    ctx.obj["input_size"] = input_size
    ctx.obj["output_size"] = output_size
    ctx.obj["hidden_size"] = hidden_size
    ctx.obj["min_hidden_size"] = min_hidden_size
    ctx.obj["use_batch_norm"] = use_batch_norm
    ctx.obj["gpu"] = gpu
    ctx.obj["criterion"] = create_criterion(criterion)
    ctx.obj["data_dir"] = data_dir

    console.log(f"[blue]Setting up model and data loaders ...[/blue]")
    
    model, _, _, test_dataloaders, criterion, _, device = setup_model_data(
        data_dir=ctx.obj["data_dir"],
        arch=ctx.obj["arch"],
        weights=ctx.obj["weights"],
        learning_rate=ctx.obj["learning_rate"],
        input_size=ctx.obj["input_size"],
        output_size=ctx.obj["output_size"],
        hidden_size=ctx.obj["hidden_size"],
        min_hidden_size=ctx.obj["min_hidden_size"],
        pretrained=ctx.obj["pretrained"],
        use_batch_norm=ctx.obj["use_batch_norm"],
        dropout=ctx.obj["dropout"],
        gpu=ctx.obj["gpu"],
        shuffle=ctx.obj["shuffle"],
        criterion=ctx.obj["criterion"],
        batch_size=ctx.obj["batch_size"]
    )
    console.log(f"[blue]Validating model on test dataset...[/blue]")
    model.eval()
    accuracy, test_loss = run_validation(model, test_dataloaders, criterion, device)
    console.log(f"Validation Accuracy: {accuracy / len(test_dataloaders):.2f}")
    console.log(f"Validation Test Loss: {test_loss / len(test_dataloaders):.3f}")

@app.command()
def train(ctx: typer.Context
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

@app.command()
def summary(ctx: typer.Context):
    """
        Display model summary.

        Usage:\n
            python -m cli summary
    """
    model, _, _, _, _, _, _, _ = setup_model_data(ctx)
    if ctx.obj["input_size"] is None or ctx.obj["input_size"] <= 0:
        ctx.obj["input_size"] = model.classifier[0].in_features

    if ctx.obj["output_size"] is None:
        ctx.obj["output_size"] = len(os.listdir(os.path.join(ctx.obj["data_dir"], 'train')))

    console.log(Classifier(
        output_size=ctx.obj["output_size"], 
        input_size=ctx.obj['input_size'],
        hidden_size=ctx.obj['hidden_size'] ,
        min_hidden_size=ctx.obj['min_hidden_size'],
        use_batch_norm=ctx.obj['use_batch_norm'],
        dropout=ctx.obj['dropout']
    ).get_architecture_summary())

@app.command()
def save(
    ctx: typer.Context,
    save_dir: Annotated[str, 
        typer.Option(...,
            "--save-dir", "-s", 
            metavar="", 
            help="Directory to save saves",
            exists=True,
            dir_okay=True
        )
    ],
):
    """
    Save the trained model checkpoint to a file.
    Args:\n
        save_dir: Directory to save the checkpoint file (example, 'checkpoints')
    Usage:\n
        python -m cli save --save-dir <save_dir>\n
        python -m cli save -s <save_dir>
    """
    model, _, _, _, _, _, _, _ = setup_model_data(ctx)
    
    if ctx.obj["input_size"] is None or ctx.obj["input_size"] <= 0:
        ctx.obj["input_size"] = model.classifier[0].in_features

    if ctx.obj["output_size"] is None:
        ctx.obj["output_size"] = len(os.listdir(os.path.join(ctx.obj["data_dir"], 'train')))

    if ctx.obj["save_dir"] is None or not ctx.obj["save_dir"]:
        ctx.obj["save_dir"] = save_dir

    checkpoint = Checkpoint(
        input_size=ctx.obj["input_size"],
        hidden_size=ctx.obj["hidden_size"],
        output_size=ctx.obj["output_size"],
    )

    checkpoint.save_checkpoint(ctx.obj["save_dir"])
    console.print(checkpoint)

@app.command()
def load(ctx: typer.Context):

    """Load a Pretrained Model from a file.

    Args:\n
        file_path: Path to the checkpoint file (example, 'checkpoint.pth')

    Usage:\n
        python -m cli load --file-path <file_path>\n
        python -m cli load -f <file_path>
    """
    console.log("Loading checkpoint...")

    checkpoints = Checkpoint(
        input_size=ctx.obj["input_size"],
        hidden_size=ctx.obj["hidden_size"],
        output_size=ctx.obj["output_size"],
    )

    try:
        model = checkpoints.load_checkpoint(file_path=ctx.obj["file_path"], gpu=ctx.obj["gpu"])
    except Exception as e:
        console.print_exception(show_locals=True)
        console.print(f"Error loading checkpoint: {e}")
        return
    console.log("Checkpoint loaded.")
    console.log(f"Model architecture: {model}")

@app.command()
def process(
    ctx: typer.Context,
    data_dir: Annotated[str, 
        typer.Option(
            "--data-dir", 
            "-d", 
            metavar="", 
            help="Path to image(s) or list of paths"
        )] = None,
    size: Annotated[Optional[int], 
        typer.Option(
            "--size", 
            "-sz", 
            metavar="", 
            help="Size for random resized crop (e.g., 224)"
        )
    ] = 224,
    mode: Annotated[Optional[str], 
        typer.Option(
            "--mode", 
            "-m", 
            metavar="", 
            help="Mode: train or inference"
        )
    ] = "inference",
    resize: Annotated[Optional[int], 
        typer.Option(
            "--resize", 
            "-rs", 
            metavar="", 
            help="Resize shortest side to this value (e.g., 256)"
        )
    ] = None,
    mean: Annotated[List[float], 
        typer.Option(
            "--mean", 
            "-m", 
            metavar="", 
            help="Normalization mean for each channel"
        )
    ] = [0.485, 0.485, 0.485],
    std: Annotated[List[float], 
        typer.Option(
            "--std", 
            "-st", 
            metavar="", 
            help="Normalization std for each channel"
        )
    ] = [0.229, 0.224, 0.225], 
    batch_size: Annotated[int, 
        typer.Option(
            "--batch-size", 
            "-b", 
            metavar="", 
            help="Batch size for dataloaders"
        )
    ] = 32,
    shuffle: Annotated[bool, 
        typer.Option(
            "--shuffle", 
            "-sh",
            metavar="", 
            help="Shuffle data in dataloaders"
        )
    ] = False
):
    """
        Preprocess images for training or inference.\n

        Usage:\n
            >>> python -m imc process --data-dir <data_dir> --size <size> --batch-size <batch_size> --shuffle --mode inference
            >>> python -m imc process --data-dir flowers --size 224 --batch-size 32 --shuffle --mode train
       \n 
       Returns PyTorch DataLoader with processed images\n
    """
    ctx.obj["data_dir"] = data_dir
    ctx.obj["size"] = size
    ctx.obj["resize"] = resize
    ctx.obj["mode"] = mode
    ctx.obj["mean"] = mean
    ctx.obj["std"] = std
    ctx.obj["batch_size"] = batch_size
    ctx.obj["shuffle"] = shuffle

    dataloader = process_image(
        image_path=ctx.obj["data_dir"],
        mode=ctx.obj["mode"],
        size=ctx.obj["size"],
        resize=ctx.obj["resize"],
        mean=ctx.obj["mean"],
        std=ctx.obj["std"],
        batch_size=ctx.obj["batch_size"],
        shuffle=ctx.obj["shuffle"]
    )

    console.log(f"[green]Processed {len(dataloader.dataset)} images into DataLoader.[/green]")
    return dataloader

@app.command()
def predict():
    """Predict the class (or classes) of an image using a trained deep learning model.
    """
    console.log("Predict command not yet implemented.")
    pass