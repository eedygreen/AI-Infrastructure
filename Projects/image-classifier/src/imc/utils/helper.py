
import os
import torch
from pathlib import Path
from typing import Optional, Dict
from rich.console import Console
from imc.middleware import Classifier

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
        class_to_idx: Optional[Dict[str, int]] = None
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
            class_to_idx: Classes mapping from indices
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
        self.class_to_idx = class_to_idx if class_to_idx is not None else {}

    def save_checkpoint(self, save_dir: str, class_to_idx: Optional[Dict[str, int]], epoch: Optional[int]=None) -> None:
        """
        Save the trained model as checkpoint.pth file.

        Args:
            file_path: Directory path to save the checkpoint
            epoch: Optional epoch number to include

        Returns:
            None
        """
        os.makedirs(save_dir, exist_ok=True)

        mapping = class_to_idx if class_to_idx is not None else self.class_to_idx
        
        checkpoint = {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_size': self.hidden_size,
            'min_hidden_size': self.min_hidden_size,
            'use_batch_norm': self.use_batch_norm,
            'dropout': self.dropout,
            'state_dict': self.state_dict(),
            'class_to_idx': mapping
        }

        if checkpoint['class_to_idx'] is None:
            console.print(f"empty class_to_idx: {checkpoint['class_to_idx']}")

        if self.optimizer is not None:
            checkpoint['optimizer'] = self.optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        save_path = os.path.join(save_dir, 'checkpoint.pth')
        
        torch.save(checkpoint, save_path)

        console.print(f"[yellow]Saving with {len(mapping)} class mappings[/yellow]")
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

        checkpoint_data = torch.load(file_path, encoding='ascii', map_location=device)

        model = cls(
            input_size = checkpoint_data['input_size'],
            output_size = checkpoint_data['output_size'],
            hidden_size = checkpoint_data['hidden_size'],
            dropout = checkpoint_data.get('dropout', 0.3),
            min_hidden_size = checkpoint_data.get('min_hidden_size', 64),
            use_batch_norm = checkpoint_data.get('use_batch_norm', True)
        )

        model.load_state_dict(checkpoint_data['state_dict'])

        model.to(device)

        console.print(f"[green]✓[/green] Model loaded successfully")
        console.print(f" Device: {device}")
        console.print(f"Architecture: {model.get_architecture_summary()}")

        if 'optimizer_state' in checkpoint_data:
            console.print("[yellow]Note[/yellow] Optimizer state available in checkpoint")
        
        return model
    