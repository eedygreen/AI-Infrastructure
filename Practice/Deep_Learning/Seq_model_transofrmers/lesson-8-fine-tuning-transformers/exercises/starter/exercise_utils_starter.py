"""
Helper utilities for Fine-Tuning Exercise
Course 3: Sequence Models and Transformers - Skill Pair 8

STUDENT VERSION - Complete the TODOs
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd


def create_squad_subset(dataset, n_samples: int, seed: int = 42):
    """
    Create a deterministic subset of SQuAD dataset.
    Filters out impossible questions (no answer).
    """
    # Filter impossible questions
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
    """
    
    # Tokenize the questions and contexts    
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
    
    # Extract mapping information
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    # Initialize positions lists
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    # Process each tokenized example to find answer positions
    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        # Get sequence IDs to identify context vs question
        sequence_ids = tokenized_examples.sequence_ids(i)
        
        # Handle examples with no answer       
        if len(answers["answer_start"]) == 0 or len(answers["text"]) == 0:
            # TODO: Append 0 to start_positions and end_positions
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
            continue
        
        # Get answer character positions
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        
        # Find context boundaries in tokenized sequence
        # Find where context starts (sequence_id == 1)
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        
        # Find where context ends
        token_end_index = len(tokenized_examples["input_ids"][i]) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1
        
        # Check if answer is in this chunk
        # Verify the answer is not truncated by checking if:
        #   offsets[token_start_index][0] <= start_char AND
        #   offsets[token_end_index][1] >= end_char
        
        if not (offsets[token_start_index][0] <= start_char and 
                offsets[token_end_index][1] >= end_char):  # TODO: Add condition here
            # Answer is outside chunk
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
        else:
            # Find exact token positions for the answer
            # Move token_start_index to the first token of the answer
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            
            # Move token_end_index to the last token of the answer
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
    
    return tokenized_examples


def prepare_validation_features(examples, tokenizer, max_length=384, doc_stride=128):
    """
    Tokenize questions and contexts for validation/testing.
    Similar to prepare_train_features but keeps example IDs.
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
    
    # Keep example IDs for mapping predictions back
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
        
        feature_indices = features_per_example.get(example_id, [])
        valid_answers = []
        
        for feature_index in feature_indices:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            
            start_indexes = np.argsort(start_logit)[-n_best_size:].tolist()
            end_indexes = np.argsort(end_logit)[-n_best_size:].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                        or end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    answer_text = context[start_char:end_char]
                    score = start_logit[start_index] + end_logit[end_index]
                    
                    valid_answers.append({
                        "text": answer_text,
                        "score": score
                    })
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            predictions_dict[example_id] = best_answer["text"]
        else:
            predictions_dict[example_id] = ""
    
    return predictions_dict


def compute_exact_match(predictions: Dict, references: Dict) -> float:
    """
    Compute Exact Match (EM) score.
    """
    em_scores = []
    
    for example_id, pred_text in predictions.items():
        if example_id in references:
            ref_texts = references[example_id]
            
            # Normalize prediction
            pred_normalized = pred_text.lower().strip()
            
            #  Normalize references
            ref_normalized = [ref.lower().strip() for ref in ref_texts]
            
            # Check if prediction matches any reference
            score = 1.0 if pred_normalized in ref_normalized else 0.0
            
            em_scores.append(score)
    
    return np.mean(em_scores) * 100 if em_scores else 0.0


def compute_f1_score(predictions: Dict, references: Dict) -> float:
    """
    Compute token-level F1 score.
    """
    f1_scores = []
    
    for example_id, pred_text in predictions.items():
        if example_id in references:
            ref_texts = references[example_id]
            
            # Tokenize prediction
            pred_tokens = pred_text.lower().split()
            
            best_f1 = 0.0
            for ref_text in ref_texts:
                # Tokenize reference
                ref_tokens = ref_text.lower().split()
                
                # Find common tokens
                common_tokens = set(pred_tokens) & set(ref_tokens)
                
                if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                    f1 = 0.0
                else:
                    # Calculate precision and recall
                    precision = len(common_tokens) / len(pred_tokens)
                    recall = len(common_tokens) / len(ref_tokens)
                    
                    # Calculate F1
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0.0
                
                best_f1 = max(best_f1, f1)
            
            f1_scores.append(best_f1)
    
    return np.mean(f1_scores) * 100 if f1_scores else 0.0


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                            title: str = "Model Performance Comparison"):
    """Create a bar chart comparing metrics across different models."""
    models = list(metrics_dict.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    em_scores = [metrics_dict[model]['EM'] for model in models]
    f1_scores = [metrics_dict[model]['F1'] for model in models]
    
    bars1 = ax.bar(x - width/2, em_scores, width, label='Exact Match (EM)', 
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', 
                   color='#e74c3c', alpha=0.8)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    return fig


def plot_learning_rate_comparison(lr_results: List[Dict]):
    """Create a comparison plot for different learning rates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    lrs = [r['lr'] for r in lr_results]
    ems = [r['em'] for r in lr_results]
    f1s = [r['f1'] for r in lr_results]
    epochs = [r['epochs'] for r in lr_results]
    
    # EM comparison
    bars1 = ax1.bar(range(len(lrs)), ems, color=['#3498db', '#e74c3c'], alpha=0.8)
    ax1.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Exact Match (%)', fontsize=12, fontweight='bold')
    ax1.set_title('EM Score by Learning Rate', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(lrs)))
    ax1.set_xticklabels([f'{lr}\n({e} epochs)' for lr, e in zip(lrs, epochs)])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 100)
    
    for bar, val in zip(bars1, ems):
        ax1.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.1f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # F1 comparison
    bars2 = ax2.bar(range(len(lrs)), f1s, color=['#3498db', '#e74c3c'], alpha=0.8)
    ax2.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Score by Learning Rate', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(lrs)))
    ax2.set_xticklabels([f'{lr}\n({e} epochs)' for lr, e in zip(lrs, epochs)])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 100)
    
    for bar, val in zip(bars2, f1s):
        ax2.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.1f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


def print_metrics_summary(baseline_metrics: Dict[str, float], 
                         finetuned_metrics: Dict[str, float]):
    """Print a formatted summary of metrics comparison."""
    print("\n" + "="*70)
    print("METRICS SUMMARY: Baseline vs. Fine-Tuned")
    print("="*70)
    
    print("\n📊 Baseline Model (Pretrained, No Fine-Tuning):")
    print(f"   • Exact Match (EM): {baseline_metrics['EM']:.1f}%")
    print(f"   • F1 Score:         {baseline_metrics['F1']:.1f}%")
    
    print("\n🎯 Fine-Tuned Model:")
    print(f"   • Exact Match (EM): {finetuned_metrics['EM']:.1f}%")
    print(f"   • F1 Score:         {finetuned_metrics['F1']:.1f}%")
    
    em_improvement = finetuned_metrics['EM'] - baseline_metrics['EM']
    f1_improvement = finetuned_metrics['F1'] - baseline_metrics['F1']
    
    print("\n📈 Improvement:")
    print(f"   • EM improved by:   {em_improvement:+.1f} percentage points")
    print(f"   • F1 improved by:   {f1_improvement:+.1f} percentage points")
    
    print("\n💡 Key Insight:")
    print(f"   Fine-tuning improved F1 by {f1_improvement:.1f} percentage points!")
    print("   This demonstrates the power of transfer learning with limited data.")
    print("="*70 + "\n")
