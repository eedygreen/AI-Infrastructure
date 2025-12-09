
import os
import torch
from pathlib import Path
from typing import Optional
from rich.console import Console
from middleware import Classifier

console = Console()

class Checkpoint(Classifier):
    """
        Checkpoint class that extends Classifier with save/load functionality.
        
        This class inherits from Classifier and can access all parent attributes.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        min_hidden_size: int = 64,
        use_batch_norm: bool = True,
        dropout: float = 0.3,
        optimizer: Optional[torch.optim.Optimizer] = None,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None
    ):
        """
            Initialize Checkpoint with all necessary parameters.
        
        Args:
            input_size: Size of input features
            hidden_size: Starting hidden layer size
            num_classes: Number of output classes
            min_hidden_size: Minimum hidden layer size
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            optimizer: The optimizer used for training
            train_dataloader: Training Dataloader (extract class_to_idx from dataset)
        """
        super().__init__(
            input_size = input_size,
            output_size = output_size,
            hidden_size = hidden_size,
            min_hidden_size = min_hidden_size,
            dropout = dropout,
            use_batch_norm = use_batch_norm
        )

        self.optimizer = optimizer

        if train_dataloader is not None:
            self.class_to_idx = train_dataloader.dataset.class_to_idx
        else:
            self.class_to_idx = {}

    def save_checkpoint(self, save_dir: str, epoch: Optional[int]=None) -> None:
        """
        Save the trained model as checkpoint.pth file.

        Args:
            file_path: Directory path to save the checkpoint
            epoch: Optional epoch number to include

        Returns:
            None
        """
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_size': self.hidden_size,
            'min_hidden_size': self.min_hidden_size,
            'use_batch_norm': self.use_batch_norm,
            'dropout': self.dropout,
            'state_dict': self.state_dict(),
            'class_to_idx': self.class_to_idx
        }

        if self.optimizer is not None:
            checkpoint['optimizer'] = self.optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        save_path = os.path.join(save_dir, 'checkpoint.pth')
        console.log("Saving checkpoint...")
        torch.save(checkpoint, save_path)
        console.print(f"[green]✓[/green] Checkpoint saved to: {save_path}")

    @classmethod
    def load_checkpoint(cls, file_path: str, gpu: Optional[bool]):
        """
        Load a checkpoint from a file.

        Args:
            file_path: Path to the checkpoint file (example, 'saves/checkpoint.pth')

            model: Pretrained Network (example vgg)

            gpu: Optional device to tremap storage location

        Returns:
            model: network with a pretrained weights
        """
        device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
        
        file_path = Path(file_path) #/ 'checkpoint.pth'

        if not file_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {file_path}")
        
        console.print(f"[blue]Loading checkpoint from:[/blue] {file_path}")

        checkpoint_data = torch.load(file_path, map_location=device)
    
        model = cls(
            input_size = checkpoint_data['input_size'],
            output_size = checkpoint_data['output_size'],
            hidden_size = checkpoint_data['hidden_size'],
            dropout = checkpoint_data.get('dropout', 0.3),
            min_hidden_size = checkpoint_data.get('min_hidden_size', 64),
            use_batch_norm = checkpoint_data.get('use_batch_norm', True)
        )

        model.load_state_dict(checkpoint_data['state_dict'])

        model.class_to_idx = checkpoint_data.get('class_to_idx', {})

        model.to(device)

        console.print(f"[green]✓[/green] Model loaded successfully")
        console.print(f" Device: {device}")
        console.print(f"Architecture: {model.get_architecture_summary()}")

        if 'optimizer_state' in checkpoint_data:
            console.print("[yellow]Note[/yellow] Optimizer state available in checkpoint")
        
        return model
    