"""
Helper utilities for Fine-Tuning DistilBERT on SQuAD 2.0
Educational demo for Udacity Deep Learning Nanodegree

FIXED VERSION: Corrected prepare_train_features function to properly
identify answer token positions using sequence_ids().
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
from typing import Dict, List, Tuple


def create_squad_subset(dataset, n_samples: int, seed: int = 42, filter_impossible: bool = True):
    """
    Create a deterministic subset of SQuAD dataset.
    
    Args:
        dataset: HuggingFace dataset object
        n_samples: Number of samples to select
        seed: Random seed for reproducibility
        filter_impossible: If True, remove questions with no answer (SQuAD 2.0)
    
    Returns:
        Filtered and subsampled dataset
    """
    # Filter impossible questions if requested
    if filter_impossible:
        dataset = dataset.filter(lambda x: len(x['answers']['text']) > 0)
    
    # Create deterministic subset
    indices = list(range(len(dataset)))
    np.random.seed(seed)
    np.random.shuffle(indices)
    selected_indices = indices[:n_samples]
    
    return dataset.select(selected_indices)


def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    """
    Tokenize questions and contexts for training.
    Handles long contexts with sliding window approach.
    
    FIXED: Now properly identifies answer span positions using sequence_ids()
    to distinguish between question tokens, context tokens, and special tokens.
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        doc_stride: Stride for sliding window on long contexts
    
    Returns:
        Tokenized features with answer positions
    """
    # Tokenize questions and contexts
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",  # Only truncate context, not question
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # Map back to original examples (needed for sliding window)
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    # Initialize start and end positions
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        # Get the example this tokenized chunk corresponds to
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        # If no answer, set cls_index as answer (model learns to predict CLS for no-answer)
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
            continue
            
        # Get answer start and end character positions
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        
        # Use sequence_ids to find where context tokens are
        # sequence_ids: None = special token, 0 = question, 1 = context
        sequence_ids = tokenized_examples.sequence_ids(i)
        
        # Find the start and end of the context in this tokenized chunk
        context_start = 0
        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1
        
        context_end = len(sequence_ids) - 1
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1
        
        # Check if context was found
        if context_start > context_end:
            # No context tokens in this chunk (shouldn't happen, but be safe)
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
            continue
        
        # Check if answer is within this chunk's context span
        # offsets[context_start][0] is the char position where context starts
        # offsets[context_end][1] is the char position where context ends
        if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
            # Answer is not fully contained in this chunk - use CLS token
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
            continue
        
        # Find the token index for answer start
        # Move forward until we find the token that contains start_char
        token_start_index = context_start
        while token_start_index <= context_end and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        token_start_index -= 1  # Back up one since we went one past
        
        # Find the token index for answer end
        # Move backward until we find the token that contains end_char
        token_end_index = context_end
        while token_end_index >= context_start and offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        token_end_index += 1  # Move forward one since we went one past
        
        # Validate the positions
        if token_start_index < context_start:
            token_start_index = context_start
        if token_end_index > context_end:
            token_end_index = context_end
        if token_start_index > token_end_index:
            # Invalid span - fall back to CLS
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
            continue
            
        tokenized_examples["start_positions"].append(token_start_index)
        tokenized_examples["end_positions"].append(token_end_index)
    
    return tokenized_examples


def prepare_validation_features(examples, tokenizer, max_length=384, doc_stride=128):
    """
    Tokenize questions and contexts for validation/testing.
    Similar to training but keeps example IDs for mapping predictions back.
    """
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # Keep example IDs and offset mapping for post-processing
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []
    
    for i in range(len(tokenized_examples["input_ids"])):
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
    
    return tokenized_examples


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30,
):
    """
    Post-process raw model predictions to extract answer spans.
    
    Args:
        examples: Original examples
        features: Tokenized features
        predictions: Tuple of (start_logits, end_logits)
        n_best_size: Number of best predictions to consider
        max_answer_length: Maximum answer length in tokens
    
    Returns:
        Dictionary mapping example IDs to predicted answers
    """
    start_logits, end_logits = predictions
    
    # Map features back to examples
    example_id_to_index = {ex["id"]: i for i, ex in enumerate(examples)}
    features_per_example = {}
    for i, feature in enumerate(features):
        example_id = feature["example_id"]
        if example_id not in features_per_example:
            features_per_example[example_id] = []
        features_per_example[example_id].append(i)
    
    # Collect predictions
    predictions_dict = {}
    
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        
        # Get all features for this example
        feature_indices = features_per_example[example_id]
        
        # Collect valid start/end positions
        valid_answers = []
        
        for feature_index in feature_indices:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            
            # Get top n_best start and end positions
            start_indexes = np.argsort(start_logit)[-n_best_size:].tolist()
            end_indexes = np.argsort(end_logit)[-n_best_size:].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip invalid answers
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                        or end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    
                    # Check for (0, 0) offsets (special tokens)
                    if offset_mapping[start_index] == (0, 0) or offset_mapping[end_index] == (0, 0):
                        continue
                    
                    # Get answer text from context
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    answer_text = context[start_char:end_char]
                    
                    # Calculate score
                    score = start_logit[start_index] + end_logit[end_index]
                    
                    valid_answers.append({
                        "text": answer_text,
                        "score": score
                    })
        
        # Select best answer
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            predictions_dict[example_id] = best_answer["text"]
        else:
            predictions_dict[example_id] = ""
    
    return predictions_dict


def compute_exact_match(predictions: Dict, references: Dict) -> float:
    """
    Compute Exact Match (EM) score.
    EM = 1 if predicted answer exactly matches any ground truth answer, 0 otherwise.
    """
    em_scores = []
    for example_id, pred_text in predictions.items():
        if example_id in references:
            ref_texts = references[example_id]
            # Normalize: lowercase and strip
            pred_normalized = pred_text.lower().strip()
            ref_normalized = [ref.lower().strip() for ref in ref_texts]
            em_scores.append(1.0 if pred_normalized in ref_normalized else 0.0)
    
    return np.mean(em_scores) if em_scores else 0.0


def compute_f1_score(predictions: Dict, references: Dict) -> float:
    """
    Compute token-level F1 score.
    Measures word overlap between prediction and ground truth.
    """
    f1_scores = []
    
    for example_id, pred_text in predictions.items():
        if example_id in references:
            ref_texts = references[example_id]
            
            # Get tokens
            pred_tokens = pred_text.lower().split()
            
            # Find best F1 among all reference answers
            best_f1 = 0.0
            for ref_text in ref_texts:
                ref_tokens = ref_text.lower().split()
                
                # Calculate overlap
                common_tokens = set(pred_tokens) & set(ref_tokens)
                
                if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                    f1 = 0.0
                else:
                    precision = len(common_tokens) / len(pred_tokens)
                    recall = len(common_tokens) / len(ref_tokens)
                    
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0.0
                
                best_f1 = max(best_f1, f1)
            
            f1_scores.append(best_f1)
    
    return np.mean(f1_scores) if f1_scores else 0.0


def visualize_model_architecture(model):
    """
    Create a high-level visualization of the model architecture.
    Shows pretrained encoder + Q&A head.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.distilbert.parameters())
    head_params = total_params - encoder_params
    
    # Draw architecture
    boxes = [
        {'name': 'Input\n[Question] [SEP] [Context]', 'y': 0.1, 'color': '#E8F4F8'},
        {'name': f'DistilBERT Encoder\n(Pretrained)\n{encoder_params:,} params', 'y': 0.35, 'color': '#B3D9E6'},
        {'name': f'Q&A Head\n(Task-Specific)\n{head_params:,} params', 'y': 0.65, 'color': '#FFA07A'},
        {'name': 'Output\n[Start Position, End Position]', 'y': 0.9, 'color': '#FFE4B5'},
    ]
    
    for i, box in enumerate(boxes):
        rect = plt.Rectangle((0.2, box['y']), 0.6, 0.15, 
                            facecolor=box['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, box['y'] + 0.075, box['name'], 
               ha='center', va='center', fontsize=11, weight='bold')
        
        # Draw arrows between boxes
        if i < len(boxes) - 1:
            ax.arrow(0.5, box['y'] + 0.15, 0, 0.05, 
                    head_width=0.05, head_length=0.02, fc='black', ec='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('DistilBERT for Question Answering Architecture', fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def plot_training_metrics(trainer_state):
    """
    Visualize training metrics from trainer state.
    Shows loss curves over training steps.
    """
    log_history = trainer_state.log_history
    
    # Extract metrics
    train_loss = []
    train_steps = []
    eval_loss = []
    eval_steps = []
    
    for entry in log_history:
        if 'loss' in entry:
            train_loss.append(entry['loss'])
            train_steps.append(entry['step'])
        if 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
            eval_steps.append(entry['step'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if train_loss:
        ax.plot(train_steps, train_loss, 'b-', label='Training Loss', linewidth=2)
    if eval_loss:
        ax.plot(eval_steps, eval_loss, 'r--', label='Validation Loss', linewidth=2, marker='o')
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress', fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_predictions(examples, baseline_preds, finetuned_preds, n_examples=3):
    """
    Create a visual comparison of baseline vs fine-tuned predictions.
    """
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 4*n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for idx, (example, ax) in enumerate(zip(examples[:n_examples], axes)):
        example_id = example['id']
        question = example['question']
        context = example['context'][:200] + '...' if len(example['context']) > 200 else example['context']
        ground_truth = example['answers']['text'][0] if example['answers']['text'] else 'No answer'
        
        baseline_answer = baseline_preds.get(example_id, "No prediction")
        finetuned_answer = finetuned_preds.get(example_id, "No prediction")
        
        ax.axis('off')
        
        # Create text display
        text = f"Example {idx+1}\n\n"
        text += f"Question: {question}\n\n"
        text += f"Context: {context}\n\n"
        text += f"Ground Truth: {ground_truth}\n"
        text += f"Baseline (Pretrained): {baseline_answer}\n"
        text += f"Fine-Tuned: {finetuned_answer}\n"
        
        # Color code correctness
        baseline_correct = baseline_answer.lower().strip() == ground_truth.lower().strip()
        finetuned_correct = finetuned_answer.lower().strip() == ground_truth.lower().strip()
        
        ax.text(0.05, 0.5, text, fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Add correctness indicators
        if finetuned_correct:
            ax.text(0.95, 0.5, '✓ Correct!', fontsize=14, color='green',
                   weight='bold', ha='right', va='center')
        elif baseline_correct:
            ax.text(0.95, 0.5, '✗ Degraded', fontsize=14, color='orange',
                   weight='bold', ha='right', va='center')
        else:
            improvement = len(set(finetuned_answer.lower().split()) & 
                            set(ground_truth.lower().split())) > \
                         len(set(baseline_answer.lower().split()) & 
                            set(ground_truth.lower().split()))
            if improvement:
                ax.text(0.95, 0.5, '↑ Improved', fontsize=14, color='blue',
                       weight='bold', ha='right', va='center')
    
    plt.tight_layout()
    return fig


def print_training_summary(train_result, eval_results_before, eval_results_after):
    """
    Print a formatted summary of training results.
    """
    print("\n" + "="*70)
    print("FINE-TUNING RESULTS SUMMARY")
    print("="*70)
    
    print("\n📊 Training Statistics:")
    print(f"  • Total training time: {train_result.metrics['train_runtime']:.1f} seconds")
    print(f"  • Training samples per second: {train_result.metrics['train_samples_per_second']:.2f}")
    print(f"  • Final training loss: {train_result.metrics['train_loss']:.4f}")
    
    print("\n🎯 Performance Improvement:")
    print(f"  Before Fine-Tuning:")
    print(f"    - Exact Match (EM): {eval_results_before['exact_match']:.1%}")
    print(f"    - F1 Score: {eval_results_before['f1']:.1%}")
    
    print(f"\n  After Fine-Tuning:")
    print(f"    - Exact Match (EM): {eval_results_after['exact_match']:.1%}")
    print(f"    - F1 Score: {eval_results_after['f1']:.1%}")
    
    print(f"\n  📈 Improvement:")
    print(f"    - EM: +{(eval_results_after['exact_match'] - eval_results_before['exact_match']):.1%}")
    print(f"    - F1: +{(eval_results_after['f1'] - eval_results_before['f1']):.1%}")
    
    print("\n" + "="*70)
    print("Key Insight: Fine-tuning on just 1,000 examples enabled the model to")
    print("learn task-specific skills while leveraging pretrained language knowledge!")
    print("="*70 + "\n")
