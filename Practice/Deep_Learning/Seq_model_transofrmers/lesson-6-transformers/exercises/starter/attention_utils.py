"""
attention_utils.py - Visualization and Analysis Tools for Transformer Attention

This module provides utilities for:
- Extracting attention weights from BERT models
- Visualizing attention heatmaps
- Comparing multi-head attention patterns
- Analyzing contextualized embeddings
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional


def plot_attention_heatmap(attention_weights, tokens, title="Attention Weights", 
                          layer=None, head=None, figsize=(10, 8), save_path=None):
    """
    Plot attention heatmap for a single attention head.
    
    Args:
        attention_weights: Attention matrix (seq_len × seq_len)
        tokens: List of token strings
        title: Plot title
        layer: Layer number (for title)
        head: Head number (for title)
        figsize: Figure size tuple
        save_path: Optional path to save figure
    
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        ax=ax,
        vmin=0,
        vmax=1,
        square=True
    )
    
    # Set title
    if layer is not None and head is not None:
        title = f"{title}\nLayer {layer}, Head {head}"
    elif layer is not None:
        title = f"{title}\nLayer {layer}"
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Key Tokens (attending TO)', fontsize=11)
    ax.set_ylabel('Query Tokens (attending FROM)', fontsize=11)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_multihead_attention(attention_weights, tokens, layer, heads=[0, 3, 7, 11],
                             figsize=(16, 12), save_path=None):
    """
    Plot 2×2 grid comparing multiple attention heads.
    
    Args:
        attention_weights: Full attention tensor from one layer (n_heads, seq_len, seq_len)
        tokens: List of token strings
        layer: Layer number
        heads: List of 4 head indices to visualize
        figsize: Figure size tuple
        save_path: Optional path to save figure
    
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, head in enumerate(heads):
        # Get attention for this head
        attn = attention_weights[head]
        
        # Create heatmap
        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            cbar_kws={'label': 'Weight'},
            ax=axes[idx],
            vmin=0,
            vmax=1,
            square=True
        )
        
        axes[idx].set_title(f'Head {head}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Key (TO)', fontsize=10)
        axes[idx].set_ylabel('Query (FROM)', fontsize=10)
        
        # Rotate labels
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].tick_params(axis='y', rotation=0)
    
    plt.suptitle(f'Multi-Head Attention Comparison - Layer {layer}', 
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_layer_progression(attention_weights_by_layer, tokens, head=0, 
                          layers=[0, 3, 7, 11], figsize=(16, 12), save_path=None):
    """
    Plot attention for one head across multiple layers (layer progression).
    
    Args:
        attention_weights_by_layer: List of attention tensors, one per layer
        tokens: List of token strings
        head: Which head to visualize
        layers: Which layers to show (0-indexed)
        figsize: Figure size tuple
        save_path: Optional path to save figure
    
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, layer in enumerate(layers):
        # Get attention for this layer and head
        attn = attention_weights_by_layer[layer][head]
        
        # Create heatmap
        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            cbar_kws={'label': 'Weight'},
            ax=axes[idx],
            vmin=0,
            vmax=1,
            square=True
        )
        
        axes[idx].set_title(f'Layer {layer+1}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Key (TO)', fontsize=10)
        axes[idx].set_ylabel('Query (FROM)', fontsize=10)
        
        # Rotate labels
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].tick_params(axis='y', rotation=0)
    
    plt.suptitle(f'Layer Progression - Head {head}', 
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compare_embeddings(embeddings, positions, word, tokens, method='cosine'):
    """
    Compare embeddings of the same word at different positions.
    
    Args:
        embeddings: Tensor of shape (seq_len, hidden_dim)
        positions: List of positions to compare
        word: The word being compared
        tokens: List of all tokens
        method: 'cosine' or 'euclidean'
    
    Returns:
        dict: Comparison results with similarities/distances
    """
    results = {
        'word': word,
        'positions': positions,
        'contexts': [tokens[pos] for pos in positions],
        'comparisons': []
    }
    
    # Extract embeddings at specified positions
    emb_list = [embeddings[pos] for pos in positions]
    
    # Compute pairwise comparisons
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            emb_i = emb_list[i]
            emb_j = emb_list[j]
            
            if method == 'cosine':
                # Cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    emb_i.unsqueeze(0), 
                    emb_j.unsqueeze(0)
                ).item()
                
                results['comparisons'].append({
                    'pos1': positions[i],
                    'pos2': positions[j],
                    'context1': tokens[positions[i]],
                    'context2': tokens[positions[j]],
                    'similarity': similarity
                })
            
            elif method == 'euclidean':
                # Euclidean distance
                distance = torch.dist(emb_i, emb_j).item()
                
                results['comparisons'].append({
                    'pos1': positions[i],
                    'pos2': positions[j],
                    'context1': tokens[positions[i]],
                    'context2': tokens[positions[j]],
                    'distance': distance
                })
    
    return results


def extract_attention_patterns(attention_weights, tokens, top_k=3):
    """
    Extract and summarize top attention patterns for each token.
    
    Args:
        attention_weights: Attention matrix (seq_len × seq_len)
        tokens: List of token strings
        top_k: Number of top attended tokens to show per query
    
    Returns:
        dict: Attention patterns for each token
    """
    patterns = {}
    
    for i, query_token in enumerate(tokens):
        # Get attention weights for this query token
        attn_row = attention_weights[i]
        
        # Get top-k attended tokens
        top_indices = np.argsort(attn_row)[-top_k:][::-1]
        top_weights = attn_row[top_indices]
        top_tokens = [tokens[idx] for idx in top_indices]
        
        patterns[query_token] = {
            'position': i,
            'top_attended': list(zip(top_tokens, top_indices, top_weights))
        }
    
    return patterns


def print_attention_patterns(patterns, max_tokens=10):
    """
    Pretty-print attention patterns.
    
    Args:
        patterns: Output from extract_attention_patterns
        max_tokens: Maximum number of tokens to display
    """
    print("=" * 80)
    print("ATTENTION PATTERNS (Top-3 attended tokens per query)")
    print("=" * 80)
    
    count = 0
    for query_token, info in patterns.items():
        if count >= max_tokens:
            print(f"\n... ({len(patterns) - max_tokens} more tokens)")
            break
        
        print(f"\n'{query_token}' (pos {info['position']}) attends to:")
        for token, pos, weight in info['top_attended']:
            print(f"  • '{token}' (pos {pos}): {weight:.3f}")
        
        count += 1


def print_model_architecture(model, detailed=False):
    """
    Pretty-print BERT model architecture.
    
    Args:
        model: HuggingFace BERT model
        detailed: Whether to print detailed layer information
    """
    print("=" * 80)
    print("BERT MODEL ARCHITECTURE")
    print("=" * 80)
    
    # Get config
    config = model.config
    
    print(f"\nModel: {config.model_type.upper()}")
    print(f"Architecture: {model.__class__.__name__}")
    
    print(f"\n📊 Key Parameters:")
    print(f"  • Vocabulary size:    {config.vocab_size:,}")
    print(f"  • Hidden size:        {config.hidden_size}")
    print(f"  • Number of layers:   {config.num_hidden_layers}")
    print(f"  • Attention heads:    {config.num_attention_heads}")
    print(f"  • Intermediate size:  {config.intermediate_size}")
    print(f"  • Max position:       {config.max_position_embeddings}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🔢 Parameters:")
    print(f"  • Total:      {total_params:,}")
    print(f"  • Trainable:  {trainable_params:,}")
    print(f"  • Size:       ~{total_params / 1e6:.1f}M parameters")
    
    if detailed:
        print(f"\n🏗️  Layer Structure:")
        print(f"  Embeddings:")
        print(f"    ├── Token Embeddings:     {config.vocab_size} × {config.hidden_size}")
        print(f"    ├── Position Embeddings:  {config.max_position_embeddings} × {config.hidden_size}")
        print(f"    └── Layer Normalization")
        
        print(f"\n  Transformer Layers (×{config.num_hidden_layers}):")
        print(f"    ├── Multi-Head Attention:")
        print(f"    │   ├── Heads: {config.num_attention_heads}")
        print(f"    │   ├── Head dim: {config.hidden_size // config.num_attention_heads}")
        print(f"    │   └── Output: {config.hidden_size}")
        print(f"    ├── Feed-Forward:")
        print(f"    │   ├── Intermediate: {config.intermediate_size}")
        print(f"    │   └── Output: {config.hidden_size}")
        print(f"    └── Layer Normalization (×2)")
        
        print(f"\n  Output:")
        print(f"    └── Pooler: {config.hidden_size} → {config.hidden_size}")
    
    print("\n" + "=" * 80)


def visualize_embedding_comparison(embeddings, positions, tokens, word, 
                                   figsize=(10, 6), save_path=None):
    """
    Visualize embedding comparison using PCA or t-SNE.
    
    Args:
        embeddings: Tensor of shape (seq_len, hidden_dim)
        positions: List of positions of the same word
        tokens: List of all tokens
        word: The word being compared
        figsize: Figure size
        save_path: Optional path to save figure
    
    Returns:
        fig: Matplotlib figure
    """
    from sklearn.decomposition import PCA
    
    # Extract embeddings
    emb_array = embeddings.cpu().numpy()
    
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb_array)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all tokens in gray
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c='lightgray', alpha=0.5, s=50)
    
    # Highlight the target word at different positions
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for idx, pos in enumerate(positions):
        color = colors[idx % len(colors)]
        ax.scatter(emb_2d[pos, 0], emb_2d[pos, 1], 
                  c=color, s=200, marker='*', 
                  edgecolors='black', linewidth=2,
                  label=f"'{word}' at pos {pos}")
        
        # Add annotation
        context = tokens[max(0, pos-2):min(len(tokens), pos+3)]
        context_str = ' '.join(context)
        ax.annotate(f"...{context_str}...", 
                   xy=(emb_2d[pos, 0], emb_2d[pos, 1]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.5', 
                   facecolor=color, alpha=0.3))
    
    ax.set_title(f'Contextualized Embeddings: "{word}" in Different Contexts', 
                fontsize=13, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("Attention Utilities Module")
    print("=" * 80)
    print("\nAvailable functions:")
    print("  • plot_attention_heatmap() - Single attention heatmap")
    print("  • plot_multihead_attention() - Compare 4 heads side-by-side")
    print("  • plot_layer_progression() - Same head across layers")
    print("  • compare_embeddings() - Compare contextualized embeddings")
    print("  • extract_attention_patterns() - Get top-k patterns")
    print("  • print_attention_patterns() - Display patterns")
    print("  • print_model_architecture() - Show BERT structure")
    print("  • visualize_embedding_comparison() - PCA visualization")
    print("\n✓ Import this module in your notebook!")
