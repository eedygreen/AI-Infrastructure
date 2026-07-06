from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    """
        A Neural Network Classifier with Batch Normalization and Dropout.

        The network progressively reduces the layer sizes from the starting hidden layer
        size down to the output layer size, applying ReLU activation, batch normalization,
        and dropout regularization at each layer.

        Args:
            input_size (int): The size of the input features (default is 25088 for VGG16).
            output_size (int): The number of output classes (default is 102 for flower classification).
            hidden_layers (int): Starting hidden size defining the sizes of the hidden layers.
            min_hidden_size (int): Minimum size for the hidden layers (default is 64).
            use_batch_norm (bool): Whether to use batch normalization after each hidden layer (default is True).
            dropout (float): The dropout probability to apply after each hidden layer (default is 0.5).

        Usage:
            # create layers of [512, 256, 128, 64] units
            model = Classifier(
                input_size=25088,
                output_size=num_classes.int,
                hidden_size=512,
                min_hidden_size=64,
                use_batch_norm=True,
                dropout=0.3
            )
    """
    def __init__(
            self,
            output_size: int,
            input_size: int,
            hidden_size: int,
            min_hidden_size: int,
            use_batch_norm: bool,
            dropout: float
        ):
        super().__init__()

        if hidden_size < min_hidden_size:
            raise ValueError(f"hidden_size must be at least {min_hidden_size}")

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1]")
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        self.min_hidden_size = min_hidden_size

        self.hidden_layers = self._generate_hidden_layers(hidden_size, min_hidden_size)

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        layer_sizes = [input_size] + self.hidden_layers

        for i in range(len(layer_sizes) - 1):
            # Linear layers
            self.layers.append(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )

            # Batch Normalization layer
            if use_batch_norm:
                self.batch_norms.append(
                    nn.BatchNorm1d(layer_sizes[i + 1])
                )

            # Dropout layer   
            self.dropouts.append(
                nn.Dropout(p=dropout)
            )

        # output layer
        self.output_layer = nn.Linear(self.hidden_layers[-1], self.output_size)
    
    def _generate_hidden_layers(self, start_size: int, min_size: int) -> List[int]:
        """
            Generate hidden layer sizes by progressively halving.

            Args:
                start_size (int): Starting size of the first hidden layer.
                min_size (int): Minimum size for the hidden layers.
            
            Returns:
                List[int]: A list of hidden layer sizes.

            Usage:
                hidden_layers = self._generate_hidden_layers(512, 64)
                # hidden_layers will be [512, 256, 128, 64]
        """
        layers = []
        current_size = start_size

        while current_size >= min_size:
            layers.append(current_size)
            if current_size == min_size:
                break
            next_size = max(current_size // 2, min_size)
            if next_size == current_size:
                break
            current_size = next_size

        return layers
    
    # Define the feed-forwarder
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass through the network.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, input_size).
                
            Returns:
                torch.Tensor: Output tensor after passing through the network.
        
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))

            if self.use_batch_norm:
                x = self.batch_norms[i](x)

            # apply dropout
            x = self.dropouts[i](x)

        # output layer
        x = self.output_layer(x)

        return F.log_softmax(x, dim=1)
    
    def get_architecture_summary(self) -> str:
        """
        Get a summary of the Classifier architecture.

        Returns:
            str: A string representation of the Classifier architecture.
        """
        summary = [
            f"Classifier Architecture Summary:",
            f"Input Size: {self.input_size}",
            f"Hidden Layers: {','.join(map(str, self.hidden_layers))}",
            f"Output Size: {self.output_size}",
            f"Dropout: {self.dropout}", 
            f"Batch Normalization: {'Enabled' if self.use_batch_norm else 'Disabled'}"
        ]

        return "\n".join(summary)
