"""
Encoder-Decoder Q&A Model Architecture.
"""
import random
import torch
import torch.nn as nn

from data import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_TOKEN


class Encoder(nn.Module):
    """
    Bidirectional LSTM Encoder.
    
    Processes (context + SEP + question) and compresses information into
    a fixed-size context vector - THE BOTTLENECK!
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len) with token indices
        
        Returns:
            hidden: Final context vector (1, batch_size, hidden_dim)
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
    
    Generates answer tokens autoregressively using hidden state from encoder
    (the compressed context vector).
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx=PAD_TOKEN):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # Unidirectional LSTM (for autoregressive generation)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden, cell):
        """
        Args:
            x: Input tensor (batch_size, seq_len) - currently generated tokens
            hidden: Hidden state from encoder or previous decoder step
            cell: Cell state from encoder or previous decoder step
        
        Returns:
            prediction: Token logits (batch_size, seq_len, vocab_size)
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
    Complete Seq2Seq Model for Q&A.
    
    Combines encoder and decoder with teacher forcing during training.
    """
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: Source tokens (batch_size, src_len) - context + SEP + question
            trg: Target tokens (batch_size, trg_len) - answer
            teacher_forcing_ratio: Probability of using ground truth tokens vs predictions
        
        Returns:
            outputs: Predictions (batch_size, trg_len, vocab_size)
        """
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features
        
        # output tensor for predictions
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        
        # This creates the BOTTLENECK: all context → single vector
        hidden, cell = self.encoder(src)
        
        # SOS token (first token of target)
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # TODO: Decode autoregressively
        # For each timestep:
        #   1. Pass decoder_input through decoder
        #   2. Store predictions
        #   3. Decide on next input: ground truth (teacher forcing) or prediction
        for t in range(trg_len):
            # TODO: Get decoder output
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t:t+1, :] = output
            
            if t < trg_len - 1:
                # TODO: Implement teacher forcing decision
                # With probability teacher_forcing_ratio, use ground truth
                # Otherwise use model prediction (argmax)
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                decoder_input = trg[:, t+1].unsqueeze(1) if use_teacher_forcing else output.argmax(2)
        
        return outputs


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
