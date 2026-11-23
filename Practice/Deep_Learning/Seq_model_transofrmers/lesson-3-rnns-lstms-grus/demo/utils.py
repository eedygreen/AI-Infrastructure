"""
Utility functions and model classes for Sequential Memory Demo
Keeps implementation details separate from the main demonstration notebook
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# DATA PROCESSING FUNCTIONS
# ========================

def build_sequences(text, seq_len, char2idx):
    """Convert text to sequence pairs for training"""
    X, y = [], []
    for i in range(len(text) - seq_len):
        sequence = [char2idx[ch] for ch in text[i:i+seq_len]]
        target = char2idx[text[i+seq_len]]
        X.append(sequence)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


def train_test_split(X, y, test_ratio=0.2):
    """Split data into training and test sets"""
    n_test = int(len(X) * test_ratio)
    n_train = len(X) - n_test
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def prepare_data(text, short_seq_len=10, long_seq_len=50):
    """Complete data preparation pipeline"""
    # Character mapping
    chars = sorted(set(text))
    vocab_size = len(chars)
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    
    # Build sequences
    short_X, short_y = build_sequences(text, short_seq_len, char2idx)
    long_X, long_y = build_sequences(text, long_seq_len, char2idx)
    
    # Train/test splits
    short_X_train, short_X_test, short_y_train, short_y_test = train_test_split(short_X, short_y)
    long_X_train, long_X_test, long_y_train, long_y_test = train_test_split(long_X, long_y)
    
    return {
        'vocab_size': vocab_size,
        'char2idx': char2idx,
        'idx2char': idx2char,
        'short_train': (short_X_train, short_y_train),
        'short_test': (short_X_test, short_y_test),
        'long_train': (long_X_train, long_y_train),
        'long_test': (long_X_test, long_y_test)
    }


# ========================
# MODEL ARCHITECTURES
# ========================

class MLP(nn.Module):
    """Feedforward network - no sequence memory - deliberately constrained"""
    def __init__(self, vocab_size, seq_len, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.flatten_size = seq_len * embed_dim
        self.layers = nn.Sequential(
            nn.Linear(self.flatten_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  # Heavy dropout to limit memorization
            nn.Linear(hidden_dim, vocab_size)
        )
        
    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.size(0), -1)  # Flatten sequence
        return self.layers(x)


class SimpleRNN(nn.Module):
    """Vanilla RNN - sequential processing with hidden state"""
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, h=None):
        embedded = self.embed(x)
        rnn_out, h_new = self.rnn(embedded, h)
        final_output = self.output_layer(rnn_out[:, -1, :])
        return final_output, h_new
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim).to(device)


class LSTMModel(nn.Module):
    """LSTM with 3 gates: forget, input, output"""
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embed(x)
        lstm_out, hidden_new = self.lstm(embedded, hidden)
        final_output = self.output_layer(lstm_out[:, -1, :])
        return final_output, hidden_new
    
    def init_hidden(self, batch_size):
        h = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return (h, c)


class GRUModel(nn.Module):
    """GRU with 2 gates: reset, update"""
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embed(x)
        gru_out, hidden_new = self.gru(embedded, hidden)
        final_output = self.output_layer(gru_out[:, -1, :])
        return final_output, hidden_new
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim).to(device)


# ========================
# TRAINING & EVALUATION
# ========================

def train_model(model, X, y, epochs=8, batch_size=64, is_sequential=True, lr=0.002):
    """Universal training function with configurable learning rate"""
    model.train()
    
    # Different learning rates for different models
    if is_sequential:
        optimizer = optim.Adam(model.parameters(), lr=lr*1.5)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr*0.8)
    
    criterion = nn.CrossEntropyLoss()
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        # Mini-batch training
        for i in range(0, len(X) - batch_size, batch_size):
            batch_x = X[i:i+batch_size].to(device)
            batch_y = y[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            
            if is_sequential:
                hidden = model.init_hidden(batch_x.size(0))
                outputs, _ = model(batch_x, hidden)
            else:
                outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping for RNNs
            if is_sequential:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        losses.append(avg_loss)
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.4f}")
    
    return losses


def evaluate_model(model, X_test, y_test, is_sequential=True):
    """Evaluate model on test set - out of sample performance"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(X_test) - batch_size, batch_size):
            batch_x = X_test[i:i+batch_size].to(device)
            batch_y = y_test[i:i+batch_size].to(device)
            
            if is_sequential:
                hidden = model.init_hidden(batch_x.size(0))
                outputs, _ = model(batch_x, hidden)
            else:
                outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == batch_y).sum().item()
            total_samples += batch_y.size(0)
    
    avg_loss = total_loss / (len(X_test) // batch_size)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def generate_sample(model, char2idx, idx2char, seed="The meaning of", max_length=100, vocab_size=None):
    """Generate text using trained model"""
    model.eval()
    chars_generated = list(seed)
    current_sequence = [char2idx.get(ch, 0) for ch in seed]
    
    with torch.no_grad():
        hidden = model.init_hidden(1)
        
        for _ in range(max_length):
            # Use last 50 characters as context
            context = current_sequence[-50:] if len(current_sequence) > 50 else current_sequence
            input_tensor = torch.tensor([context]).to(device)
            
            output, hidden = model(input_tensor, hidden)
            probabilities = torch.softmax(output, dim=1)
            next_char_idx = torch.multinomial(probabilities, 1).item()
            
            if vocab_size and next_char_idx < vocab_size:
                next_char = idx2char[next_char_idx]
                chars_generated.append(next_char)
                current_sequence.append(next_char_idx)
    
    return ''.join(chars_generated)


# ========================
# UTILITY FUNCTIONS
# ========================

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters())


def create_all_models(vocab_size):
    """Create all model instances for comparison"""
    models = {
        'mlp_short': MLP(vocab_size, 10).to(device),
        'mlp_long': MLP(vocab_size, 50).to(device),
        'rnn': SimpleRNN(vocab_size).to(device),
        'lstm': LSTMModel(vocab_size).to(device),
        'gru': GRUModel(vocab_size).to(device)
    }
    
    # Print parameter counts
    for name, model in models.items():
        print(f"{name.upper():<10} params: {count_parameters(model):,}")
    
    return models


def print_performance_summary(results):
    """Print a formatted performance summary"""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    for model_name, (test_loss, test_acc) in results.items():
        print(f"{model_name:<12}: Loss = {test_loss:.4f}, Accuracy = {test_acc:.3f}")


def analyze_parameter_fairness(models):
    """Analyze parameter counts for fair comparison"""
    print("\n" + "="*60)
    print("PARAMETER ANALYSIS")
    print("="*60)
    
    for name, model in models.items():
        params = count_parameters(model)
        print(f"{name:<12}: {params:,} parameters")
    
    # Check for unfair advantages
    mlp_long_params = count_parameters(models['mlp_long'])
    rnn_params = count_parameters(models['rnn'])
    
    if mlp_long_params > rnn_params * 1.2:
        print("\n⚠️  WARNING: MLP has significantly more parameters!")
        print("   This may give unfair advantage in direct comparison")
    else:
        print("\n✅ Parameter counts are reasonably balanced")
