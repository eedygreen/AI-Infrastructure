"""
attention_viz.py - Attention Visualization Utilities

Functions for visualizing attention weights as heatmaps.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_attention(attention_weights, source_tokens, target_tokens, title="Attention Weights"):
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention_weights: Attention matrix (target_len, source_len)
        source_tokens: List of source words
        target_tokens: List of target words
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        vmin=0,
        vmax=1
    )
    
    ax.set_xlabel('Source Words (Input)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target Words (Output)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    return fig


def plot_attention_simple(attention_weights, source_text, target_text, title="Attention Visualization"):
    """
    Simplified attention plot with text strings.
    
    Args:
        attention_weights: Attention matrix (target_len, source_len) as numpy array
        source_text: Source sentence as string
        target_text: Target sentence as string
        title: Plot title
    """
    source_tokens = source_text.split()
    target_tokens = target_text.split()
    
    return plot_attention(attention_weights, source_tokens, target_tokens, title)


def plot_attention_comparison(attn_weights_list, source_tokens_list, target_tokens_list, 
                              titles, overall_title="Attention Comparison"):
    """
    Plot multiple attention heatmaps side by side for comparison.
    
    Args:
        attn_weights_list: List of attention matrices
        source_tokens_list: List of source token lists
        target_tokens_list: List of target token lists
        titles: List of subplot titles
        overall_title: Overall figure title
    """
    n_plots = len(attn_weights_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots, 6))
    
    if n_plots == 1:
        axes = [axes]
    
    for i, (attn, src, tgt, title) in enumerate(zip(
        attn_weights_list, source_tokens_list, target_tokens_list, titles
    )):
        sns.heatmap(
            attn,
            xticklabels=src,
            yticklabels=tgt,
            cmap='YlOrRd',
            cbar_kws={'label': 'Weight'},
            linewidths=0.5,
            linecolor='gray',
            ax=axes[i],
            vmin=0,
            vmax=1
        )
        
        axes[i].set_xlabel('Source Words', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Target Words', fontsize=11, fontweight='bold')
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        
        axes[i].tick_params(axis='x', rotation=45)
    
    fig.suptitle(overall_title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def highlight_max_attention(attention_weights, source_tokens, target_tokens):
    """
    Print a text summary showing which source words each target word focuses on most.
    
    Args:
        attention_weights: Attention matrix (target_len, source_len)
        source_tokens: List of source words
        target_tokens: List of target words
    """
    print("Attention Focus Summary:")
    print("=" * 60)
    
    for t_idx, target_word in enumerate(target_tokens):
        src_idx = np.argmax(attention_weights[t_idx])
        max_weight = attention_weights[t_idx, src_idx]
        
        print(f"'{target_word}' → focuses on '{source_tokens[src_idx]}' "
              f"(weight: {max_weight:.3f})")
    
    print("=" * 60)


def plot_attention_for_example(model, context, question, answer, vocab, idx2word, device):
    """
    Generate and plot attention for a single Q&A example.
    
    Args:
        model: Seq2SeqWithAttention model
        context: Context string
        question: Question string
        answer: Answer string (ground truth)
        vocab: Vocabulary dict
        idx2word: Index to word dict
        device: torch device
        
    Returns:
        figure: Matplotlib figure
        attention_weights: Attention matrix as numpy array
    """
    import torch
    from data import encode_text, SEP_TOKEN, EOS_TOKEN, SOS_TOKEN, PAD_TOKEN
    
    model.eval()
    
    with torch.no_grad():
        # Encode input
        ctx_ids = encode_text(context, vocab)
        q_ids = encode_text(question, vocab)
        src_ids = (ctx_ids + [SEP_TOKEN] + q_ids + [EOS_TOKEN])[:100]
        src_ids += [PAD_TOKEN] * (100 - len(src_ids))
        src = torch.LongTensor([src_ids]).to(device)
        
        # Encode answer for target
        ans_ids = encode_text(answer, vocab)
        trg_ids = [SOS_TOKEN] + ans_ids + [EOS_TOKEN]
        trg_ids += [PAD_TOKEN] * (100 - len(trg_ids))
        trg = torch.LongTensor([trg_ids[:100]]).to(device)
        
        # Forward pass
        outputs, attentions = model(src, trg, teacher_forcing_ratio=0)
        
        # Get attention weights (first batch item)
        attn = attentions[0].cpu().numpy()  # (trg_len, src_len)
        
        # Get actual lengths
        src_words = context.split() + ['<SEP>'] + question.split()
        ans_words = answer.split()
        
        # Trim attention to actual lengths
        attn = attn[:len(ans_words), :len(src_words)]
        
        # Plot
        fig = plot_attention(attn, src_words, ans_words, 
                           title=f"Attention: '{question}' → '{answer}'")
        
        return fig, attn


if __name__ == "__main__":
    # Test visualization
    print("Testing attention visualization utilities")
    
    # Create sample attention weights
    attention = np.array([
        [0.1, 0.8, 0.1],
        [0.2, 0.1, 0.7],
        [0.7, 0.2, 0.1]
    ])
    
    source = ["Alice", "lives", "Paris"]
    target = ["in", "Paris", "city"]
    
    # Test plot
    fig = plot_attention(attention, source, target, "Test Attention")
    plt.savefig('/tmp/test_attention.png')
    print("✓ Test plot saved to /tmp/test_attention.png")
    
    # Test summary
    highlight_max_attention(attention, source, target)
