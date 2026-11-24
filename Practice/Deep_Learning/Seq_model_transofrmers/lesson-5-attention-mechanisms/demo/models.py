"""
models.py - Seq2Seq Models for Q&A Exercise

This module contains the encoder-decoder architecture for the Q&A task.
Includes both basic seq2seq and attention-based versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    """
    Bidirectional LSTM Encoder.
    
    Processes input sequence and produces context vector (the bottleneck!).
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Linear layers to combine bidirectional states
        # From hidden_dim*2 (forward + backward) to hidden_dim
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor (batch_size, seq_len)
            
        Returns:
            hidden: Final hidden state (1, batch_size, hidden_dim)
            cell: Final cell state (1, batch_size, hidden_dim)
        """
        # Embed input
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # Pass through bidirectional LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        # hidden: (2, batch_size, hidden_dim) - 2 for bidirectional
        # outputs: (batch_size, seq_len, hidden_dim * 2)
        
        # Combine forward (hidden[-2]) and backward (hidden[-1]) directions
        hidden = torch.tanh(
            self.fc_hidden(torch.cat([hidden[-2], hidden[-1]], dim=1))
        ).unsqueeze(0)
        
        cell = torch.tanh(
            self.fc_cell(torch.cat([cell[-2], cell[-1]], dim=1))
        ).unsqueeze(0)
        
        return hidden, cell


class Decoder(nn.Module):
    """
    LSTM Decoder.
    
    Generates output sequence autoregressively from context vector.
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx=0):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # Unidirectional LSTM (for autoregressive generation)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden, cell):
        """
        Forward pass through decoder.
        
        Args:
            x: Input tensor (batch_size, 1) - one token at a time
            hidden: Hidden state from encoder/previous step
            cell: Cell state from encoder/previous step
            
        Returns:
            prediction: Output logits (batch_size, 1, vocab_size)
            hidden: Updated hidden state
            cell: Updated cell state
        """
        # Embed input
        embedded = self.embedding(x)  # (batch, 1, embed_dim)
        
        # Pass through LSTM
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: (batch_size, 1, hidden_dim)
        
        # Project to vocabulary
        prediction = self.fc(output)  # (batch_size, 1, vocab_size)
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """
    Complete Sequence-to-Sequence Model.
    
    Combines encoder and decoder with teacher forcing support.
    """
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass through seq2seq model.
        
        Args:
            src: Source sequence (batch_size, src_len)
            trg: Target sequence (batch_size, trg_len) 
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Predictions (batch_size, trg_len, vocab_size)
        """
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        
        # Encode source sequence into context vector
        # THIS IS THE BOTTLENECK: All source info compressed into hidden/cell
        hidden, cell = self.encoder(src)
        
        # First decoder input is first token of target
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # Decode autoregressively
        for t in range(trg_len):
            # Generate prediction for position t
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t:t+1, :] = output
            
            # Prepare input for next position
            if t < trg_len - 1:
                # Decide whether to use teacher forcing
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                
                if use_teacher_forcing:
                    # Use ground truth token (teacher forcing)
                    decoder_input = trg[:, t+1].unsqueeze(1)
                else:
                    # Use model's own prediction
                    decoder_input = output.argmax(2)
        
        return outputs


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.
    
    Computes attention weights between decoder state and all encoder states.
    This eliminates the fixed-size bottleneck!
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Attention network components
        # Input: hidden (hidden_dim) + encoder_outputs (hidden_dim*2 from bidirectional) = hidden_dim*3
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        """
        Compute attention weights and context vector.
        
        Args:
            hidden: Decoder hidden state (batch_size, hidden_dim)
            encoder_outputs: All encoder hidden states (batch_size, src_len, hidden_dim*2)
            
        Returns:
            context: Weighted sum of encoder outputs (batch_size, hidden_dim*2)
            attention_weights: Attention distribution (batch_size, src_len)
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden: (batch_size, src_len, hidden_dim)
        
        # Concatenate decoder hidden with each encoder output
        combined = torch.cat((hidden, encoder_outputs), dim=2)
        # combined: (batch_size, src_len, hidden_dim*3)
        
        # Compute attention scores (energy)
        energy = torch.tanh(self.attn(combined))
        # energy: (batch_size, src_len, hidden_dim)
        
        # Project to scalar score for each position
        attention_scores = self.v(energy).squeeze(2)
        # attention_scores: (batch_size, src_len)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        # attention_weights: (batch_size, src_len)
        
        # Compute context vector as weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        # context: (batch_size, 1, hidden_dim*2)
        
        context = context.squeeze(1)
        # context: (batch_size, hidden_dim*2)
        
        return context, attention_weights


class AttentionDecoder(nn.Module):
    """
    LSTM Decoder with Bahdanau Attention.
    
    Uses attention to dynamically focus on relevant encoder states.
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx=0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # Attention mechanism
        self.attention = BahdanauAttention(hidden_dim)
        
        # LSTM takes embedding + context vector as input
        self.lstm = nn.LSTM(embed_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        
        # Output projection from hidden_dim to vocab
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden, cell, encoder_outputs):
        """
        Forward pass with attention.
        
        Args:
            x: Input tensor (batch_size, 1)
            hidden: Hidden state (1, batch_size, hidden_dim)
            cell: Cell state (1, batch_size, hidden_dim)
            encoder_outputs: All encoder states (batch_size, src_len, hidden_dim*2)
            
        Returns:
            prediction: Output logits (batch_size, 1, vocab_size)
            hidden: Updated hidden state
            cell: Updated cell state
            attention_weights: Attention distribution (batch_size, src_len)
        """
        # Embed input
        embedded = self.embedding(x)  # (batch_size, 1, embed_dim)
        
        # Compute attention
        # hidden is (1, batch_size, hidden_dim), need (batch_size, hidden_dim)
        context, attention_weights = self.attention(hidden.squeeze(0), encoder_outputs)
        # context: (batch_size, hidden_dim*2)
        # attention_weights: (batch_size, src_len)
        
        # Concatenate embedded input with attention context
        lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        # lstm_input: (batch_size, 1, embed_dim + hidden_dim*2)
        
        # Pass through LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: (batch_size, 1, hidden_dim)
        
        # Project to vocabulary
        prediction = self.fc(output)  # (batch_size, 1, vocab_size)
        
        return prediction, hidden, cell, attention_weights


class Seq2SeqWithAttention(nn.Module):
    """
    Sequence-to-Sequence Model with Attention.
    
    Eliminates the fixed-size bottleneck by using attention mechanism.
    """
    
    def __init__(self, encoder, attention_decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = attention_decoder
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass with attention.
        
        Args:
            src: Source sequence (batch_size, src_len)
            trg: Target sequence (batch_size, trg_len)
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Predictions (batch_size, trg_len, vocab_size)
            attention_weights: All attention weights (batch_size, trg_len, src_len)
        """
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features
        src_len = src.shape[1]
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        
        # Tensor to store attention weights
        attentions = torch.zeros(batch_size, trg_len, src_len).to(src.device)
        
        # Encode source sequence
        # Get both final state AND all encoder outputs
        embedded = self.encoder.embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder.lstm(embedded)
        # encoder_outputs: (batch_size, src_len, hidden_dim*2) - ALL states!
        
        # Combine bidirectional hidden/cell states
        hidden = torch.tanh(
            self.encoder.fc_hidden(torch.cat([hidden[-2], hidden[-1]], dim=1))
        ).unsqueeze(0)
        cell = torch.tanh(
            self.encoder.fc_cell(torch.cat([cell[-2], cell[-1]], dim=1))
        ).unsqueeze(0)
        
        # First decoder input
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # Decode with attention
        for t in range(trg_len):
            # Generate prediction for position t with attention
            output, hidden, cell, attention_weights = self.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            
            outputs[:, t:t+1, :] = output
            attentions[:, t, :] = attention_weights
            
            # Prepare input for next position
            if t < trg_len - 1:
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                
                if use_teacher_forcing:
                    decoder_input = trg[:, t+1].unsqueeze(1)
                else:
                    decoder_input = output.argmax(2)
        
        return outputs, attentions


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    print("Testing Seq2Seq Models")
    print("="*80)
    
    # Hyperparameters
    vocab_size = 100
    embed_dim = 128
    hidden_dim = 256
    batch_size = 4
    src_len = 20
    trg_len = 10
    
    # Create models
    encoder = Encoder(vocab_size, embed_dim, hidden_dim)
    decoder = Decoder(vocab_size, embed_dim, hidden_dim)
    model = Seq2Seq(encoder, decoder)
    
    print(f"Model created with {count_parameters(model):,} parameters")
    print(f"Context vector size: {hidden_dim} dimensions")
    
    # Test forward pass
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    trg = torch.randint(0, vocab_size, (batch_size, trg_len))
    
    output = model(src, trg, teacher_forcing_ratio=0.5)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {src.shape}")
    print(f"  Target shape: {trg.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  ✓ Model working correctly!")
