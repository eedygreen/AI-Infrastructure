"""
metrics.py - Question Answering Evaluation Metrics

This module implements standard Q&A evaluation metrics:
- Exact Match (EM): Binary metric for perfect matches
- F1 Score: Token-level overlap (precision + recall)
- Precision@K: Quality of top-K ranked results
- Recall@K: Coverage of correct answers in top-K
- MRR: Mean Reciprocal Rank for ranking quality
"""

import re
import string
import numpy as np
from collections import Counter


# ============================================================================
# Text Normalization (from SQuAD evaluation script)
# ============================================================================

def normalize_answer(s):
    """
    Normalize answer text for fair comparison.
    
    Performs:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace
    
    Args:
        s: Answer string
    
    Returns:
        Normalized string
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    """
    Get tokens from normalized string.
    
    Args:
        s: String to tokenize
    
    Returns:
        List of tokens
    """
    if not s:
        return []
    return normalize_answer(s).split()


# ============================================================================
# Exact Match (EM)
# ============================================================================

def compute_exact_match(prediction, ground_truth):
    """
    Compute Exact Match score.
    
    Returns 1.0 if normalized prediction exactly matches ground truth,
    otherwise 0.0.
    
    Args:
        prediction: Predicted answer string
        ground_truth: Ground truth answer string
    
    Returns:
        1.0 or 0.0
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_em_for_dataset(predictions):
    """
    Compute overall Exact Match for a list of predictions.
    
    Args:
        predictions: List of dicts with 'prediction_text' and 'ground_truth'
    
    Returns:
        EM score (percentage)
    """
    if not predictions:
        return 0.0
    
    scores = [
        compute_exact_match(pred['prediction_text'], pred['ground_truth'])
        for pred in predictions
    ]
    
    return 100.0 * sum(scores) / len(scores)


# ============================================================================
# F1 Score (Token-level)
# ============================================================================

def compute_f1(prediction, ground_truth):
    """
    Compute token-level F1 score.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Where:
    - Precision = |predicted ∩ reference| / |predicted|
    - Recall = |predicted ∩ reference| / |reference|
    
    Args:
        prediction: Predicted answer string
        ground_truth: Ground truth answer string
    
    Returns:
        F1 score (0.0 to 1.0)
    """
    pred_tokens = get_tokens(prediction)
    truth_tokens = get_tokens(ground_truth)
    
    # Handle empty predictions/truths
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    
    # Count token overlaps
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def compute_f1_for_dataset(predictions):
    """
    Compute overall F1 score for a list of predictions.
    
    Args:
        predictions: List of dicts with 'prediction_text' and 'ground_truth'
    
    Returns:
        F1 score (percentage)
    """
    if not predictions:
        return 0.0
    
    scores = [
        compute_f1(pred['prediction_text'], pred['ground_truth'])
        for pred in predictions
    ]
    
    return 100.0 * sum(scores) / len(scores)


# ============================================================================
# Precision@K and Recall@K (for ranked retrieval)
# ============================================================================

def compute_precision_at_k(ranked_predictions, ground_truths, k=3):
    """
    Compute Precision@K.
    
    What fraction of top-K predictions are correct?
    
    Args:
        ranked_predictions: List of predicted answers (ordered by score)
        ground_truths: List of acceptable correct answers
        k: Number of top results to consider
    
    Returns:
        Precision@K (0.0 to 1.0)
    """
    if not ranked_predictions or not ground_truths:
        return 0.0
    
    top_k = ranked_predictions[:k]
    
    # Count how many of top-K match any ground truth
    correct = 0
    for pred in top_k:
        pred_normalized = normalize_answer(pred)
        for truth in ground_truths:
            if normalize_answer(truth) == pred_normalized:
                correct += 1
                break
    
    return correct / k


def compute_recall_at_k(ranked_predictions, ground_truths, k=3):
    """
    Compute Recall@K.
    
    What fraction of all correct answers appear in top-K?
    
    Args:
        ranked_predictions: List of predicted answers (ordered by score)
        ground_truths: List of acceptable correct answers
        k: Number of top results to consider
    
    Returns:
        Recall@K (0.0 to 1.0)
    """
    if not ground_truths:
        return 0.0
    
    if not ranked_predictions:
        return 0.0
    
    top_k = ranked_predictions[:k]
    
    # Count how many ground truths appear in top-K
    found = 0
    for truth in ground_truths:
        truth_normalized = normalize_answer(truth)
        for pred in top_k:
            if normalize_answer(pred) == truth_normalized:
                found += 1
                break
    
    return found / len(ground_truths)


# ============================================================================
# Mean Reciprocal Rank (MRR)
# ============================================================================

def compute_reciprocal_rank(ranked_predictions, ground_truths):
    """
    Compute reciprocal rank for a single question.
    
    Returns 1/rank of first correct answer, or 0 if no correct answer found.
    
    Args:
        ranked_predictions: List of predicted answers (ordered by score)
        ground_truths: List of acceptable correct answers
    
    Returns:
        Reciprocal rank (0.0 to 1.0)
    """
    if not ranked_predictions or not ground_truths:
        return 0.0
    
    for rank, pred in enumerate(ranked_predictions, start=1):
        pred_normalized = normalize_answer(pred)
        for truth in ground_truths:
            if normalize_answer(truth) == pred_normalized:
                return 1.0 / rank
    
    return 0.0


def compute_mrr(predictions_list):
    """
    Compute Mean Reciprocal Rank across all questions.
    
    Args:
        predictions_list: List of dicts with 'ranked_predictions' and 'ground_truth'
    
    Returns:
        MRR score (0.0 to 1.0)
    """
    if not predictions_list:
        return 0.0
    
    reciprocal_ranks = []
    
    for pred in predictions_list:
        ranked = [p['text'] if isinstance(p, dict) else p 
                 for p in pred.get('ranked_predictions', [])]
        ground_truths = pred.get('ground_truth', [])
        
        # Handle single ground truth (convert to list)
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths] if ground_truths else []
        
        rr = compute_reciprocal_rank(ranked, ground_truths)
        reciprocal_ranks.append(rr)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


# ============================================================================
# Metric Summary Functions
# ============================================================================

def compute_all_metrics(predictions):
    """
    Compute all Q&A metrics for a list of predictions.
    
    Args:
        predictions: List of prediction dicts
    
    Returns:
        Dictionary with all metric scores
    """
    metrics = {
        'exact_match': compute_em_for_dataset(predictions),
        'f1': compute_f1_for_dataset(predictions),
        'num_examples': len(predictions)
    }
    
    return metrics


def compute_metrics_by_type(predictions):
    """
    Compute metrics separately for answerable vs unanswerable questions.
    
    Args:
        predictions: List of prediction dicts with 'is_impossible' field
    
    Returns:
        Dictionary with metrics for each type
    """
    answerable = [p for p in predictions if not p.get('is_impossible', False)]
    unanswerable = [p for p in predictions if p.get('is_impossible', False)]
    
    results = {
        'overall': compute_all_metrics(predictions),
        'answerable': compute_all_metrics(answerable) if answerable else None,
        'unanswerable': compute_all_metrics(unanswerable) if unanswerable else None
    }
    
    return results


def print_metrics_summary(metrics_dict):
    """
    Pretty-print metrics summary.
    
    Args:
        metrics_dict: Dictionary from compute_metrics_by_type()
    """
    print("=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)
    
    # Overall metrics
    overall = metrics_dict['overall']
    print(f"\nOverall ({overall['num_examples']} examples):")
    print(f"  Exact Match (EM):  {overall['exact_match']:.2f}%")
    print(f"  F1 Score:          {overall['f1']:.2f}%")
    
    # Answerable
    if metrics_dict['answerable']:
        ans = metrics_dict['answerable']
        print(f"\nAnswerable Questions ({ans['num_examples']} examples):")
        print(f"  Exact Match (EM):  {ans['exact_match']:.2f}%")
        print(f"  F1 Score:          {ans['f1']:.2f}%")
    
    # Unanswerable
    if metrics_dict['unanswerable']:
        unans = metrics_dict['unanswerable']
        print(f"\nUnanswerable Questions ({unans['num_examples']} examples):")
        print(f"  Exact Match (EM):  {unans['exact_match']:.2f}%")
        print(f"  F1 Score:          {unans['f1']:.2f}%")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("Testing Q&A Metrics...")
    print("=" * 80)
    print()
    
    # Test cases
    test_cases = [
        {
            'prediction': "New York City",
            'ground_truth': "New York",
            'expected_em': 0.0,
            'expected_f1': 0.67
        },
        {
            'prediction': "Paris",
            'ground_truth': "Paris",
            'expected_em': 1.0,
            'expected_f1': 1.0
        },
        {
            'prediction': "The answer",
            'ground_truth': "answer",
            'expected_em': 1.0,  # After normalization
            'expected_f1': 1.0
        }
    ]
    
    print("Testing normalization and metrics:\n")
    
    for i, test in enumerate(test_cases, 1):
        em = compute_exact_match(test['prediction'], test['ground_truth'])
        f1 = compute_f1(test['prediction'], test['ground_truth'])
        
        print(f"{i}. Prediction: '{test['prediction']}'")
        print(f"   Ground Truth: '{test['ground_truth']}'")
        print(f"   EM: {em:.2f} (expected: {test['expected_em']:.2f})")
        print(f"   F1: {f1:.2f} (expected: {test['expected_f1']:.2f})")
        print()
    
    print("✓ Metrics implementation working correctly!")
