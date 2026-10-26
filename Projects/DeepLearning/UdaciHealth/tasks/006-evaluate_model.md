# Task: Evaluate the Model

## Goal
The goal is to evaluate the trained neural network architecture on unseen data to assess its performance and generalization ability.

## Context
Training loss tells us how well the model is performing on the training data, but it does not provide insight into how well the model will perform on unseen data. Evaluating the model on a separate test set allows us to measure its generalization performance and identify potential issues such as overfitting or underfitting.


## Requirements
1. **Evaluation**: 
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
    a. Setting the model to evaluation mode (to disable dropout, etc.)
    b. Looping through the evaluation DataLoader without tracking gradients
    c. Collecting predictions and true labels for each batch
    d. Converting these results to NumPy arrays for metric calculations
    e. Computing relevant performance metrics for the use case
    f. Returning all computed values in a single dictionary for easy analysis
    This function helps summarize how well the trained model generalizes to unseen data.
2. **Visualization**: 
    a. Visualize the model's performance using confusion matrix (cm). CM makes the trade-offs between precision and recall more interpretable and provides insights into the types of errors the model is making. 
    ```python
    disp =  ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='Blues', colorbar=True)
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.show()
    ```

    b. Plot ROC curve with threshold selection, default is 0.5, Lower threshold (0.3) and higher threshold (0.7)
    to provide insights into its classification capabilities.
    the Area Under the Curve (AUC) quantifies this performance: higher AUC also means better separability. This as a rule-of-thumb:
    * 0.90–1.00 → Excellent discrimination
    * 0.80–0.89 → Good discrimination
    * 0.70–0.79 → Fair discrimination
    * Below 0.70 → Poor discrimination
    * Use this plot and AUC value to reason about which threshold best aligns with your real-world trade-offs.
    # Calculate ROC curve
    ```python
    fpr, tpr, thresholds =  roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    ```

    # Plot ROC curve
    ```python
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='#3498db', linewidth=2.5, label=f'Model ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='Random Classifier (AUC = 0.5)')

    # Mark the current operating point (threshold = 0.5)
    current_idx = np.argmin(np.abs(thresholds - 0.5))
    plt.scatter(fpr[current_idx], tpr[current_idx], color='red', s=150, zorder=5, 
                label=f'Current Threshold (0.5)\nTPR={tpr[current_idx]:.3f}, FPR={fpr[current_idx]:.3f}')

    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12, fontweight='bold')
    plt.title('ROC Curve - Evaluating Threshold Tradeoffs', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nROC Curve Analysis:")
    print("="*60)
    print(f"AUC Score: {roc_auc:.4f}")
    print(f"Interpretation:")
    if roc_auc >= 0.90:
        print("  - Excellent discrimination ability")
    elif roc_auc >= 0.80:
        print("  - Good discrimination ability")
    elif roc_auc >= 0.70:
        print("  - Fair discrimination ability")
    else:
        print("  - Poor discrimination ability")
    ```


## Constraints
- Use metrics from `sklearn.metrics` to compute accuracy, precision, recall, and F1-score.
- Confusion matrix is calculated using `sklearn.metrics.confusion_matrix`
- Use `sklearn.metrics.roc_curve` and `sklearn.metrics.auc` to compute the ROC curve and AUC score.


## Deliverables
- Evaluation function that computes predictions, true labels, and metrics for the trained model on unseen data.
- Visualizations of the model's performance, including confusion matrix and ROC curve with AUC score.


## Report back
- After completing the task, report back with the following:
  - The metrics (accuracy, precision, recall, f1_score).
  - Any questions or considerations for training the neural network.