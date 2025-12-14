
import os, typer
from rich.console import Console
from middleware import Classifier
from typing_extensions import Annotated
from typing import Optional, List
from config import setup_model_data, LossFunction, create_criterion
from utils import Checkpoint, process_image, run_validation, train_model, run_predict

app = typer.Typer(no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]})
console = Console(stderr=True)

@app.callback()
def main(ctx: typer.Context,
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
    ] = True,
    batch_size: Annotated[int,
        typer.Option(
            "--batch-size",
            "-bs",
            metavar="",
            help="Batch size"
        )
    ] = 32
):
    """
    Global Parameters

    Usage:\n
        python -m imc -h \n
        python -m cli --arch <architecture> --batch-size <batch_size> --data-dir <data_dir> validate 
    """
    ctx.ensure_object(dict) 
    ctx.obj.update({
        "arch": arch,
        "weights": weights,
        "pretrained": pretrained,
        "learning_rate": learning_rate,
        "shuffle": shuffle,
        "dropout": dropout,
        "input_size": input_size,
        "output_size":  output_size,
        "hidden_size": hidden_size,
        "min_hidden_size": min_hidden_size,
        "use_batch_norm": use_batch_norm,
        "gpu": gpu,
        "criterion":create_criterion(criterion),
        "data_dir": data_dir,
        "batch_size": batch_size,
        
    })
    """
        Image Classifier application.

        Help:\n
            python -m imc -h \n
    """
    model, train_dataloaders, valid_dataloaders, test_dataloaders, criterion, optimizer, device, class_to_idx = setup_model_data(
        ctx.obj["data_dir"], 
        ctx.obj["arch"],
        ctx.obj["weights"],
        ctx.obj["learning_rate"],
        ctx.obj["input_size"],
        ctx.obj["output_size"],
        ctx.obj["hidden_size"],
        ctx.obj["min_hidden_size"],
        ctx.obj["pretrained"],
        ctx.obj["use_batch_norm"],
        ctx.obj["dropout"],
        ctx.obj["gpu"],
        ctx.obj["shuffle"],
        ctx.obj["criterion"],
        ctx.obj["batch_size"]
    )
    ctx.obj["model"] = model
    ctx.obj["train_dataloaders"] = train_dataloaders
    ctx.obj["test_dataloaders"] = test_dataloaders
    ctx.obj["valid_dataloaders"] = valid_dataloaders
    ctx.obj["criterion"] = criterion
    ctx.obj["class_to_idx"] = class_to_idx
    ctx.obj["device"] = device
    ctx.obj["optimizer"] = optimizer

@app.command()
def validate(ctx: typer.Context):
    """
    Validate the model on the test dataset.

    Usage:\n
        python -m cli validate \n
        python -m cli validate --arch <architecture> --batch-size <batch_size> --data-dir <data_dir>
    """
    ctx.ensure_object(dict)
    model = ctx.obj["model"]
    test_dataloaders = ctx.obj["test_dataloaders"]
    criterion = ctx.obj["criterion"]
    device = ctx.obj["device"]

    console.log(f"[blue]Setting up model and data loaders ...[/blue]")

    console.log(f"[blue]Validating model on test dataset...[/blue]")
    model.eval()
    accuracy, test_loss = run_validation(model, test_dataloaders, criterion, device)
    console.log(f"Validation Accuracy: {accuracy / len(test_dataloaders):.2f}")
    console.log(f"Validation Test Loss: {test_loss / len(test_dataloaders):.3f}")

@app.command()
def train(
    ctx: typer.Context,
    learning_rate: Annotated[float, 
        typer.Option(
            "--learning-rate", 
            "-lr", metavar="", 
            help="Learning rate for training"
        )
    ] = 0.001,
    epochs: Annotated[int,
        typer.Option(
            "--epochs", 
            "-e", 
            metavar="", 
            help="Number of epochs for training"
        )
    ] = 5,
    print_every: Annotated[int, 
        typer.Option(
            "--print-every", 
            "-p", metavar="", 
            help="Print training info every n steps"
        )
    ] = 5,
    save_dir: Annotated[str,
        typer.Option(
            "--save-dir", 
            "-s", 
            metavar="", 
            help="Directory to save checkpoints",
            exists=True,
            dir_okay=True
        )
    ] = None,
    gpu: Annotated[str,
        typer.Option(
            "--save-dir", 
            "-s", 
            metavar="", 
            help="Directory to save checkpoints",
            exists=True,
            dir_okay=True
        )
    ] = None
):
    """
    Train the neural network model.

    Usage:\n
        python -m cli train --epochs 5 --learning-rate 0.001 \n
        python -m cli train --epochs <epochs> --print_every <print_every>
 
    """
    ctx.ensure_object(dict) 
    ctx.obj["learning_rate"] = learning_rate
    ctx.obj["epochs"] = epochs
    ctx.obj["print_every"] = print_every
    ctx.obj["save_dir"] = save_dir
    ctx.obj["gpu"] = gpu
    console.log("Setting up model and data loaders ...")

    train_model(
        learning_rate=ctx.obj["learning_rate"],
        epochs=ctx.obj["epochs"],
        print_every=ctx.obj["print_every"],
        model = ctx.obj["model"],
        dataloaders=ctx.obj["train_dataloaders"], 
        test_dataloaders=ctx.obj["test_dataloaders"], 
        optimizer=ctx.obj["optimizer"],
        gpu=ctx.obj["gpu"],
        criterion=ctx.obj["criterion"],
        save_dir=ctx.obj["save_dir"]
    )

@app.command()
def summary(ctx: typer.Context):
    """
        Display model summary.

        Usage:\n
            python -m cli summary
    """
    ctx.ensure_object(dict)
    model = ctx.obj["model"]

    input_size = ctx.obj["input_size"]
    if input_size is None or input_size <= 0:
        input_size = model.classifier.layers[0].in_features

    output_size = ctx.obj["output_size"]
    if output_size is None:
        output_size = len(os.listdir(os.path.join(ctx.obj["data_dir"], 'train')))

    console.log(Classifier(
        output_size=output_size, 
        input_size=input_size,
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
    ctx.ensure_object(dict)

    model = ctx.obj["model"]
    class_to_idx = ctx.obj["class_to_idx"]

    input_size = ctx.obj["input_size"]
    if input_size is None or input_size <= 0:
        input_size = model.classifier.layers[0].in_features

    output_size = ctx.obj["output_size"]
    if output_size is None:
         output_size = len(os.listdir(os.path.join(ctx.obj["data_dir"], 'train')))

    checkpoint = Checkpoint(
        input_size=input_size,
        hidden_size=ctx.obj["hidden_size"],
        output_size=output_size,  
    )
   
    checkpoint.save_checkpoint(save_dir=save_dir, class_to_idx=class_to_idx)
    console.log(f"[green]✓ Checkpoint Saved Succefully![/green]")

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
    ctx.ensure_object(dict)

    ctx.obj["data_dir"] = data_dir
    ctx.obj["size"] = size
    ctx.obj["resize"] = resize
    ctx.obj["mode"] = mode
    ctx.obj["mean"] = mean
    ctx.obj["std"] = std
    ctx.obj["batch_size"] = batch_size
    ctx.obj["shuffle"] = shuffle

    dataloader, class_to_idx = process_image(
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
    console.log(f"[green]mappings: {class_to_idx}[/green]")
    return dataloader

@app.command()
def predict(
    ctx: typer.Context,
    image_dir: Annotated[str, 
        typer.Option(
            "--image-dir", 
            "-i", 
            metavar="", 
            help="Path to image directory"
        )] = None,
    model_path: Annotated[str, 
        typer.Option(
            "--model-path", 
            "-m", 
            metavar="", 
            help="Path to trained model file"
        )] = None,
    topk: Annotated[int, 
        typer.Option(
            "--topk", 
            "-k", 
            metavar="", 
            help="Number of top most likely classes to return"
        )] = 5,
    label_names: Annotated[str, 
        typer.Option(
            "--label-names", 
            "-l", 
            metavar="", 
            help="Path to JSON file mapping labels to real names"
        )] = None,
    gpu: Annotated[bool,
        typer.Option(
            "--gpu", 
            "-gp",
            metavar="",
            help="Enable GPU Device for Faster inference. Default is False (Disable)"
        )    ] = False
):
    """Predict the class (or classes) of an image using a trained deep learning model.

    Usage:\n
        python -m cli predict --image-path <"path_to_image"> --model-path <path_to_model> --topk 5 \n
        python -m cli predict --image-path <"path_to_images"> --model-path <path_to_model_checkpoint> --topk 3 --label-names cat_to_name.json \n
    """
    ctx.ensure_object(dict)

    ctx.obj["image_dir"] = image_dir
    ctx.obj["model_path"] = model_path
    ctx.obj["topk"] = topk
    ctx.obj["label_names"] = label_names
    ctx.obj["gpu"] = gpu
    
    run_predict(
        image_path=ctx.obj["image_dir"],
        checkpoint_path=ctx.obj["model_path"],
        gpu=ctx.obj["gpu"],
        topk=ctx.obj["topk"],   
        category_names=ctx.obj["label_names"],
        class_to_idx=ctx.obj["class_to_idx"]
    )