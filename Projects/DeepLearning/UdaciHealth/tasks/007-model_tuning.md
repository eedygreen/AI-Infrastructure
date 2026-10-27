# Task: Improve and Tune Model

## Goal
The goal is to improve the performance of the trained neural network model by tuning hyperparameters and making architectural adjustments if necessary based on evaluation results. This may involve experimenting with different learning rates, batch sizes, number of hidden layers, number of neurons per layer, and activation functions to achieve better accuracy, precision, recall, and F1-score on unseen data.

## Context
Based on the evaluation metrics obtained from the previous task, we can identify areas where the model may be underperforming. Hyperparameter tuning is a critical step in optimizing model performance and can significantly impact the model's ability to generalize to unseen data. This task involves systematically adjusting hyperparameters and retraining the model to find the optimal configuration. This is an iterative process that may require multiple rounds of experimentation and evaluation to achieve the desired performance metrics. The process nature is: evaluate the model, identify areas for improvement, adjust hyperparameters, retrain, and re-evaluate until satisfactory performance is achieved.


## Requirements
Before running the model tuning task, ensure that the following requirements are met:
1. **Experiment Tracking**: here is the code to track experiments and compare results:
```python
experiment_results = {}

def track_experiment(name: str, 
                     model: nn.Module,
                     train_losses: List[float],
                     val_losses: List[float],
                     test_results: Dict[str, float],
                     notes: str = "") -> None:
    """
    Track experiment results for later comparison.
    
    Args:
        name: Experiment name
        model: Trained model
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        test_results: Dictionary from evaluate_model()
        notes: Optional notes about the experiment
    """
   # Keep only numeric metrics so we don't try to tabulate arrays, dicts, etc.
    numeric_metrics = {
        k: float(v) for k, v in test_results.items()
        if isinstance(v, numbers.Number)
    }

    experiment_results[name] = {
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
        'min_val_loss': float(min(val_losses)),
        'loss_gap': float(abs(train_losses[-1] - val_losses[-1])),
        'metrics': numeric_metrics,           # store all metrics here
        'notes': notes,
        'train_losses': train_losses,         # keep full histories if desired
        'val_losses': val_losses
    }
    print(f"✓ Experiment '{name}' tracked successfully!")


def display_experiment_comparison(sort_by: Optional[str] = "f1",
                                  descending: bool = True) -> Optional[pd.DataFrame]:
    """
    Display a comparison table of all tracked experiments.

    Args:
        sort_by: Metric name to sort by (e.g., 'f1', 'roc_auc', 'accuracy').
                 If None or not present, will fall back to:
                   1) 'roc_auc' if present
                   2) any available metric (alphabetical)
        descending: Sort order for the chosen metric.

    Returns:
        DataFrame with experiment results.
    """
    if not experiment_results:
        print("No experiments tracked yet!")
        return None

    # Collect the union of all metric names across experiments
    all_metric_names = set()
    for res in experiment_results.values():
        all_metric_names.update(res.get('metrics', {}).keys())
    all_metric_names = sorted(all_metric_names)  # stable ordering

    # Determine sort metric
    chosen_sort = None
    if sort_by in all_metric_names:
        chosen_sort = sort_by
    elif all_metric_names:
        chosen_sort = all_metric_names[0]  # fallback to first available
    # If no metrics at all, we’ll sort by final_val_loss

    # Build table rows
    rows = []
    for name, res in experiment_results.items():
        row = {
            'Experiment': name,
            'Val Loss': f"{res['final_val_loss']:.4f}",
            'Loss Gap': f"{res['loss_gap']:.4f}",
        }
        # Add metrics (formatted)
        for m in all_metric_names:
            val = res['metrics'].get(m, None)
            row[m.upper() if m.islower() else m] = (f"{val:.4f}" if isinstance(val, numbers.Number) else "")
        # Also keep raw for sorting
        row['_sort_val'] = (
            res['metrics'].get(chosen_sort)
            if chosen_sort is not None else res['final_val_loss']
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort
    if chosen_sort is not None:
        df = df.sort_values('_sort_val', ascending=not descending)
    else:
        # No metrics available: sort by Val Loss ascending
        df = df.sort_values('Val Loss', ascending=True, key=lambda s: s.astype(float))

    # Clean up helper column
    df = df.drop(columns=['_sort_val'])

    return df

# Track baseline experiment
track_experiment(
    name='Baseline',
    model=model,
    train_losses=train_losses,
    val_losses=val_losses,
    test_results=test_results,
    notes='Initial model with default hyperparameters'
)

print("\nExperiment tracking system initialized!")
print("Use track_experiment() after training each variation.")
print("Use display_experiment_comparison() to see all results.")
```
2. **Systematically experiment with different hyperparameters and architectural adjustments**. For each experiment:
   - a. Define change: Specify which hyperparameter(s) or architectural change(s) you are testing (e.g., learning rate, batch size, number of hidden layers, number of neurons per layer, activation functions, dropout rate).
   - b. Train the model with the new configuration using the training function implemented in the previous task.
   - c. Evaluate the model on the validation set and test set using the evaluation function implemented in the previous task.
   - d. Track the results using the `track_experiment()` function
   - e. Visualize the training and validation loss curves for each experiment to observe convergence behavior and potential overfitting or underfitting.
   - d. Compare the results of different experiments using the `display_experiment_comparison()` function to identify the best-performing configuration based on evaluation metrics.

3. **Hyperparameter Tuning**: 
 a. **Dropout Rate**: Experiment with different dropout rates (e.g., 0.1, 0.2, 0.3) to prevent overfitting and improve generalization. here is model definition with dropout layer:
```python
class DiabetesClassifierWithDropout(nn.Module):
    def __init__(self, input_size, hidden_sizes=(BATCH_SIZE, 32), dropout=0.3):
        super().__init__()
        h1, h2 = hidden_sizes
        self.network = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)

model_dropout = DiabetesClassifierWithDropout(num_features).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model_dropout.parameters(), lr=0.001)
```
 b. **Learning Rate**: Test different learning rates (e.g., 0.01, 0.001, 0.0001) to find the optimal rate for convergence and stability during training.
    - lower lr: 0.0001
    - higher lr: 0.01
    - in between lr: 0.001
  ```python
  print("Experiment 2: Learning Rate Tuning")
  print("="*60)

  # ------- Add your code here
  ```
 c. **Adjust network architecture**: Experiment with different numbers of hidden layers and neurons per layer to find the optimal architecture for the task. For example, try adding an additional hidden layer or increasing/decreasing the number of neurons in existing layers.
  - 1 hidden layer with 32 neurons.
  - 2 hidden layers with 32 and 16 neurons.
  - 3 hidden layers with 64, 32, and 16 neurons.
  - Make the model too simple, and the model underfits.
  - Make the model too complex, and it overfits (especially with limited data).
  Compare how these choices influence parameter count, training behavior, and generalization.

 d. **Compare all experiments**: After running multiple experiments with different hyperparameters and architectures, compare the results using the `display_experiment_comparison()` function to identify the best-performing configuration based on evaluation metrics.
 ```python
 # Display comprehensive comparison of all experiments
print("Comprehensive Experiment Comparison")
print("="*70)
print("\nAll experiments sorted by F1-Score (best to worst):\n")

comparison_df = display_experiment_comparison()
print(comparison_df.to_string(index=False))

# Identify best configuration
best_experiment = comparison_df.iloc[0]['Experiment']
print(f"\n{'='*70}")
print(f"Best Configuration: {best_experiment}")
print(f"{'='*70}")

print(f"\nKey Insights:")
print("  - Review the 'Loss Gap' column to identify which techniques reduced overfitting")
print("  - Consider the tradeoff between Precision and Recall for your use case")
```
 e. **Visualize loss curves for top 3 experiments**
 ```python
print("\nVisualizing Loss Curves for Top 3 Experiments")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
top_3_experiments = comparison_df.head(3)['Experiment'].values

for idx, exp_name in enumerate(top_3_experiments):
    exp_data = experiment_results[exp_name]
    epochs = range(1, len(exp_data['train_losses']) + 1)
    
    axes[idx].plot(epochs, exp_data['train_losses'], label='Train Loss', linewidth=2, color='#3498db')
    axes[idx].plot(epochs, exp_data['val_losses'], label='Val Loss', linewidth=2, color='#e74c3c')
    axes[idx].set_xlabel('Epoch', fontsize=11)
    axes[idx].set_ylabel('Loss', fontsize=11)
    axes[idx].set_title(f'{exp_name}', fontsize=12, fontweight='bold')
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Constraints
- DO NOT use SDG at this stage. The focus is on hyperparameter tuning and architectural adjustments to improve model performance.


## Deliverables
- A series of experiments with different hyperparameters and architectural adjustments, each tracked using the `track_experiment()` function.
- A comprehensive comparison of all experiments using the `display_experiment_comparison()` function, highlighting the best-performing configuration based on evaluation metrics.
- Visualizations of training and validation loss curves for each experiment to observe convergence behavior and potential overfitting or underfitting.


## Report back
- After completing the task, report back with the following:
  - The metrics (accuracy, precision, recall, f1_score).
  - Any questions or considerations for training the neural network.