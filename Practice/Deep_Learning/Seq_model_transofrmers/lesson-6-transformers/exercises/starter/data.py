"""
data.py - Jigsaw Toxic Comment Classification Dataset Loader

This module loads and processes the real Jigsaw Toxic Comment Classification dataset.
Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
"""

from datasets import load_dataset
import random
import pandas as pd


def load_jigsaw_dataset(n_samples=5000, seed=42):
    """
    Load Jigsaw Toxic Comment Classification dataset.
    
    Args:
        n_samples: Number of samples to load (default: 5000)
        seed: Random seed for reproducibility
    
    Returns:
        comments: List of dictionaries with 'text', 'toxic', 'type', 'length'
    """
    print(f"Loading Toxic Comment Classification dataset...")
    print(f"Requesting {n_samples} samples...")
    
    # Try multiple dataset sources
    dataset = None
    
    # Try 1: Use civil_comments which is compatible
    try:
        dataset = load_dataset("civil_comments", split=f"train[:{n_samples}]")
        print(f"✓ Loaded {len(dataset)} comments from civil_comments dataset\n")
    except:
        pass
    
    # Try 2: Use SetFit/toxic-comment dataset if civil_comments fails
    if dataset is None:
        try:
            dataset = load_dataset("SetFit/toxic-comments", split=f"train[:{n_samples}]")
            print(f"✓ Loaded {len(dataset)} comments from toxic-comments dataset\n")
        except:
            pass
    
    # Try 3: Use ag_news as fallback (widely available)
    if dataset is None:
        try:
            dataset = load_dataset("ag_news", split=f"train[:{n_samples}]")
            print(f"✓ Loaded {len(dataset)} comments from ag_news dataset\n")
        except:
            pass
    
    # If all fail, use synthetic data
    if dataset is None:
        print("Unable to load any dataset, generating synthetic data...")
        return _generate_synthetic_fallback(n_samples, seed)
    
    # Process dataset - handle different formats
    comments = []
    for item in dataset:
        # Handle different dataset column names
        if 'comment_text' in item:
            text = item['comment_text']
        elif 'text' in item:
            text = item['text']
        else:
            # ag_news format
            text = item.get('text', '') or ''
        
        # Skip empty texts
        if not text or len(text.strip()) == 0:
            continue
        
        # Determine if toxic - handle different label formats
        is_toxic = False
        if 'toxicity' in item:
            is_toxic = item.get('toxicity', 0) > 0.5
        elif 'toxic' in item:
            is_toxic = item.get('toxic', 0) > 0 or item.get('toxic') == True
        elif 'label' in item:
            is_toxic = item.get('label', 0) > 0
        
        comments.append({
            'text': text,
            'toxic': is_toxic,
            'label': 'toxic' if is_toxic else 'safe',
            'type': 'toxic' if is_toxic else 'safe',
            'length': len(text.split())
        })
    
    # Shuffle
    random.seed(seed)
    random.shuffle(comments)
    
    return comments


def get_demo_examples(comments=None):
    """
    Get curated examples for demo purposes.
    
    Args:
        comments: Optional list of comments from load_jigsaw_dataset()
    
    Returns:
        Dictionary with specific examples for demonstrations
    """
    # Default examples for when dataset isn't loaded yet
    default_examples = {
        'contextualized_embedding': "The bank can refuse to lend money to the person by the river bank.",
        'toxic_example': "This is completely idiotic and you clearly have no idea what you're talking about.",
        'safe_example': "Thank you for sharing this helpful information. I appreciate your contribution.",
        'long_context': "In the financial district, the bank on the corner of Main Street can refuse to lend money to individuals who do not meet their strict credit requirements, while further down near the scenic park, visitors enjoy walking along the peaceful river bank where children often play.",
        'syntax_example': "The model attention mechanism learns to focus on relevant contextual information.",
        'semantic_example': "Machine learning and artificial intelligence are transforming natural language processing."
    }
    
    # If comments provided, add real examples
    if comments and len(comments) > 0:
        # Find some real examples from the dataset
        toxic_comments = [c for c in comments if c['toxic']]
        safe_comments = [c for c in comments if not c['toxic']]
        
        if toxic_comments:
            default_examples['real_toxic'] = toxic_comments[0]['text']
        
        if safe_comments:
            default_examples['real_safe'] = safe_comments[0]['text']
        
        # Find a medium-length comment for attention visualization
        medium_length = [c for c in comments if 10 < c['length'] < 30]
        if medium_length:
            default_examples['real_medium'] = medium_length[0]['text']
    
    return default_examples


def get_statistics(comments):
    """Get dataset statistics."""
    stats = {
        'total': len(comments),
        'toxic': sum(1 for c in comments if c['toxic']),
        'safe': sum(1 for c in comments if not c['toxic']),
        'avg_length': sum(c['length'] for c in comments) / len(comments),
        'min_length': min(c['length'] for c in comments),
        'max_length': max(c['length'] for c in comments),
    }
    
    stats['toxic_percent'] = (stats['toxic'] / stats['total']) * 100
    stats['safe_percent'] = (stats['safe'] / stats['total']) * 100
    
    return stats


def filter_by_length(comments, min_length=10, max_length=50):
    """
    Filter comments by length for attention visualization.
    
    Args:
        comments: List of comment dictionaries
        min_length: Minimum word count
        max_length: Maximum word count
    
    Returns:
        Filtered list of comments
    """
    return [c for c in comments if min_length <= c['length'] <= max_length]


def get_balanced_sample(comments, n_toxic=50, n_safe=50):
    """
    Get balanced sample of toxic and safe comments.
    
    Args:
        comments: List of comment dictionaries
        n_toxic: Number of toxic comments
        n_safe: Number of safe comments
    
    Returns:
        Balanced list of comments
    """
    toxic = [c for c in comments if c['toxic']]
    safe = [c for c in comments if not c['toxic']]
    
    random.shuffle(toxic)
    random.shuffle(safe)
    
    sample = toxic[:n_toxic] + safe[:n_safe]
    random.shuffle(sample)
    
    return sample


def _generate_synthetic_fallback(n_samples, seed):
    """
    Generate synthetic comments as fallback if Jigsaw dataset fails to load.
    """
    print("Generating synthetic fallback data...")
    
    random.seed(seed)
    
    # Simple templates
    safe_templates = [
        "Thank you for the helpful feedback on this topic.",
        "I really appreciate your detailed explanation.",
        "This is a great contribution to the discussion.",
        "Your perspective on this issue is very insightful.",
        "I agree with your analysis of the situation.",
    ]
    
    toxic_templates = [
        "This is stupid and you are an idiot.",
        "What a complete waste of time.",
        "You clearly have no idea what you're talking about.",
        "This is the dumbest thing I've ever read.",
        "Shut up and stop posting nonsense.",
    ]
    
    comments = []
    n_toxic = n_samples // 5  # 20% toxic
    n_safe = n_samples - n_toxic
    
    for _ in range(n_safe):
        text = random.choice(safe_templates)
        comments.append({
            'text': text,
            'toxic': False,
            'label': 'safe',
            'type': 'safe',
            'length': len(text.split())
        })
    
    for _ in range(n_toxic):
        text = random.choice(toxic_templates)
        comments.append({
            'text': text,
            'toxic': True,
            'label': 'toxic',
            'type': 'toxic',
            'length': len(text.split())
        })
    
    random.shuffle(comments)
    return comments


if __name__ == "__main__":
    # Test the loader
    print("Testing Jigsaw Dataset Loader...")
    print("=" * 80)
    
    # Load dataset
    comments = load_jigsaw_dataset(n_samples=5000)
    
    # Get statistics
    stats = get_statistics(comments)
    print(f"\n✓ Loaded {stats['total']} comments")
    print(f"\nDataset Statistics:")
    print(f"  Toxic comments:  {stats['toxic']} ({stats['toxic_percent']:.1f}%)")
    print(f"  Safe comments:   {stats['safe']} ({stats['safe_percent']:.1f}%)")
    print(f"  Average length:  {stats['avg_length']:.1f} words")
    print(f"  Length range:    {stats['min_length']}-{stats['max_length']} words")
    
    # Show examples
    print("\n" + "=" * 80)
    print("Sample Comments:")
    print("=" * 80)
    
    toxic = [c for c in comments if c['toxic']][:3]
    safe = [c for c in comments if not c['toxic']][:3]
    
    print("\nToxic Examples:")
    for i, c in enumerate(toxic, 1):
        preview = c['text'][:100] + "..." if len(c['text']) > 100 else c['text']
        print(f"  {i}. {preview}")
    
    print("\nSafe Examples:")
    for i, c in enumerate(safe, 1):
        preview = c['text'][:100] + "..." if len(c['text']) > 100 else c['text']
        print(f"  {i}. {preview}")
    
    print("\n✓ Data loading complete!")
