# Task: Train the Model

## Goal
The goal is to train a neural network architecture through the dataset to learn patterns and relationships between features and the target variable (diabetes risk). 


## Context
Training a neural network involves feeding the model with batches of data, computing predictions, calculating loss, and updating the model's parameters through backpropagation. The training process is iterative and requires careful monitoring to ensure convergence and avoid overfitting.



## Requirements
- **Optimizer**: Use an appropriate optimizer (Adam) to update the model's parameters based on the computed gradients. Set a learning rate that balances convergence speed and stability. optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
- **Loss Function**: Use a suitable loss function (Binary Cross-Entropy Loss) to measure the discrepancy between predicted probabilities and true labels. criterion = nn.BCELoss()
- **training loop**: Implement a training loop that iterates over the training data for a specified number of epochs. For each batch, perform the following steps:
  1. Move the input features and labels to the appropriate device (CPU or GPU).
  2. Zero the gradients of the optimizer to prevent accumulation from previous batches.
  3. Perform a forward pass through the model to obtain predictions.
  4. Compute the loss using the criterion.
  5. Perform backpropagation to compute gradients.
  6. Update the model's parameters using the optimizer.
Here is the training function to implement:
```python
def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                num_epochs: int = 100,
                print_every: int = 10) -> Tuple[List[float], List[float]]:
    """
    Train a PyTorch model with validation monitoring.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (CPU or GPU)
        num_epochs: Number of training epochs
        print_every: Print progress every N epochs
    
    Returns:
        Tuple of (num_epochs, train_losses, val_losses) - the losses contain one value per epoch
    """

    # Lists to track metrics over epochs
    train_losses = []
    val_losses = []
    
    # ------- Add your code here

    print("\nTraining completed!")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")

    return num_epochs, train_losses, val_losses
```

## Constraints
- Gradient Reset: Add a reset step for the optimizer's gradients before each backward pass to maintain correct gradient flow and prevent interference between batches.
- Set the model to training mode using model.train().
- Set the model to evaluation mode using model.eval() during validation to disable layers like dropout and batch normalization updates during evaluation.
- Iterate through validation batches without computing gradients to save memory and computation. Use torch.no_grad() context manager during validation.
- Calculate average validation loss and other relevant performance metrics


## Deliverables
- train_model function implemented and tested with the training and validation data loaders.
- LOGGING: Use the `print_every` argument to conditionally print progress (epoch index, avg train/val loss).


## Report back
- After completing the task, report back with the following:
  - The training parameters of the model.
  - The metrics (train_losses, val_losses).
  - Any questions or considerations for training the neural network.