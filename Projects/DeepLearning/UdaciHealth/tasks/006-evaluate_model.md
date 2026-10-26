# Task: Evaluate the Model

## Goal
The goal is to evaluate the trained neural network architecture on unseen data to assess its performance and generalization ability.

## Context
Training loss tells us how well the model is performing on the training data, but it does not provide insight into how well the model will perform on unseen data. Evaluating the model on a separate test set allows us to measure its generalization performance and identify potential issues such as overfitting or underfitting.


## Requirements
Implement the evaluation function that takes the trained model and test data loader as inputs and computes the following metrics:
- **Accuracy**: The proportion of correctly predicted instances out of the total instances.
- **Precision**: The proportion of true positive predictions out of all positive predictions made by the model.
- **Recall**: The proportion of true positive predictions out of all actual positive instances in the test set.
- **F1-score**: The harmonic mean of precision and recall, providing a single metric that balances both precision and recall.
```python
def evaluate_model(model: nn.Module, 
                   data_loader: DataLoader, 
                   device: torch.device,
                   threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate a trained model on a dataset and return comprehensive metrics.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader containing the evaluation dataset
        device: Device to run evaluation on (CPU or GPU)
        threshold: Decision threshold for classification (default: 0.5)
    
    Returns:
        Dictionary containing:
            - predictions: numpy array of predicted probabilities
            - true_labels: numpy array of true labels
            - pred_labels: numpy array of predicted binary labels
            - metrics
    """
    all_predictions = []
    all_labels = []
    results = []

    # ------- complete the evaluation function to compute predictions, true labels, and metrics
    
    return results
```
This function should test how well the model performs by:
1. Setting the model to evaluation mode (to disable dropout, etc.)
2. Looping through the evaluation DataLoader without tracking gradients
3. Collecting predictions and true labels for each batch
4. Converting these results to NumPy arrays for metric calculations
5. Computing relevant performance metrics for the use case
6. Returning all computed values in a single dictionary for easy analysis
This function helps summarize how well the trained model generalizes to unseen data.


## Constraints
- Use metrics from `sklearn.metrics` to compute accuracy, precision, recall, and F1-score.


## Deliverables
- Evaluation function that computes predictions, true labels, and metrics for the trained model on unseen data.


## Report back
- After completing the task, report back with the following:
  - The metrics (accuracy, precision, recall, f1_score).
  - Any questions or considerations for training the neural network.