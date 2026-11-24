"""
data.py - SQuAD 2.0 Dataset Loader

This module loads the real SQuAD 2.0 dataset for question answering evaluation.
SQuAD 2.0 includes both answerable and unanswerable questions.
"""

from datasets import load_dataset
import random


def load_squad_v2(split="validation", n_samples=1000, seed=42):
    """
    Load SQuAD 2.0 dataset.
    
    Args:
        split: Dataset split ('validation' or 'train')
        n_samples: Number of samples to load
        seed: Random seed for reproducibility
    
    Returns:
        List of dictionaries with question, context, answers, id
    """
    print(f"Loading SQuAD 2.0 {split} split...")
    print(f"Requesting {n_samples} samples...")
    
    # Load from HuggingFace datasets
    try:
        dataset = load_dataset("squad_v2", split=split)
        print(f"✓ Loaded {len(dataset)} total examples from SQuAD 2.0\n")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []
    
    # Shuffle and take subset
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    subset_indices = indices[:min(n_samples, len(dataset))]
    subset = dataset.select(subset_indices)
    
    # Convert to list of dicts
    examples = []
    for item in subset:
        examples.append({
            'id': item['id'],
            'question': item['question'],
            'context': item['context'],
            'answers': item['answers'],
            'is_impossible': len(item['answers']['text']) == 0  # Unanswerable
        })
    
    return examples


def get_statistics(examples):
    """
    Get dataset statistics.
    
    Args:
        examples: List of SQuAD examples
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total': len(examples),
        'answerable': sum(1 for ex in examples if not ex['is_impossible']),
        'unanswerable': sum(1 for ex in examples if ex['is_impossible'])
    }
    
    stats['answerable_percent'] = (stats['answerable'] / stats['total']) * 100
    stats['unanswerable_percent'] = (stats['unanswerable'] / stats['total']) * 100
    
    # Question and context lengths
    question_lengths = [len(ex['question'].split()) for ex in examples]
    context_lengths = [len(ex['context'].split()) for ex in examples]
    
    stats['avg_question_length'] = sum(question_lengths) / len(question_lengths)
    stats['avg_context_length'] = sum(context_lengths) / len(context_lengths)
    
    return stats


def get_demo_examples(examples, n_examples=5):
    """
    Get curated examples for demonstration.
    
    Args:
        examples: List of SQuAD examples
        n_examples: Number of examples to return
    
    Returns:
        List of diverse examples (answerable + unanswerable)
    """
    answerable = [ex for ex in examples if not ex['is_impossible']]
    unanswerable = [ex for ex in examples if ex['is_impossible']]
    
    demo = []
    
    # Get mix of answerable and unanswerable
    if answerable:
        demo.extend(answerable[:max(1, n_examples - 1)])
    
    if unanswerable:
        demo.append(unanswerable[0])
    
    return demo[:n_examples]


def filter_by_answer_length(examples, min_words=1, max_words=10):
    """
    Filter examples by answer length.
    
    Args:
        examples: List of SQuAD examples
        min_words: Minimum answer word count
        max_words: Maximum answer word count
    
    Returns:
        Filtered list
    """
    filtered = []
    for ex in examples:
        if ex['is_impossible']:
            continue
        
        answer = ex['answers']['text'][0] if ex['answers']['text'] else ""
        word_count = len(answer.split())
        
        if min_words <= word_count <= max_words:
            filtered.append(ex)
    
    return filtered


def separate_by_type(examples):
    """
    Separate examples into answerable and unanswerable.
    
    Args:
        examples: List of SQuAD examples
    
    Returns:
        Tuple of (answerable_examples, unanswerable_examples)
    """
    answerable = [ex for ex in examples if not ex['is_impossible']]
    unanswerable = [ex for ex in examples if ex['is_impossible']]
    
    return answerable, unanswerable


if __name__ == "__main__":
    # Test the loader
    print("Testing SQuAD 2.0 Loader...")
    print("=" * 80)
    
    # Load dataset
    examples = load_squad_v2(n_samples=100)
    
    # Get statistics
    stats = get_statistics(examples)
    print(f"\n✓ Loaded {stats['total']} examples")
    print(f"\nDataset Statistics:")
    print(f"  Answerable questions:    {stats['answerable']} ({stats['answerable_percent']:.1f}%)")
    print(f"  Unanswerable questions:  {stats['unanswerable']} ({stats['unanswerable_percent']:.1f}%)")
    print(f"  Avg question length:     {stats['avg_question_length']:.1f} words")
    print(f"  Avg context length:      {stats['avg_context_length']:.1f} words")
    
    # Show examples
    print("\n" + "=" * 80)
    print("Sample Questions:")
    print("=" * 80)
    
    demo = get_demo_examples(examples, n_examples=3)
    
    for i, ex in enumerate(demo, 1):
        print(f"\n{i}. {'[UNANSWERABLE]' if ex['is_impossible'] else '[ANSWERABLE]'}")
        print(f"   Question: {ex['question']}")
        print(f"   Context preview: {ex['context'][:100]}...")
        if not ex['is_impossible']:
            print(f"   Answer: {ex['answers']['text'][0]}")
    
    print("\n✓ Data loading complete!")
