"""
data.py - Synthetic Q&A Data Generation for Seq2Seq Exercise

This module generates controlled synthetic Q&A data designed to clearly
demonstrate the context vector bottleneck.
"""

import random
from typing import List, Tuple

# Special tokens
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
SEP_TOKEN = 4


class SyntheticQAGenerator:
    """Generate synthetic Q&A data with controlled context lengths."""
    
    def __init__(self, seed=42):
        random.seed(seed)
        
        # Building blocks for synthetic data
        self.entities = [
            'Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 
            'Grace', 'Henry', 'Iris', 'Jack', 'Kate', 'Leo', 
            'Mary', 'Nathan', 'Olivia', 'Peter', 'Quinn', 'Rachel',
            'Sam', 'Tina', 'Uma', 'Victor', 'Wendy', 'Xavier'
        ]
        
        self.locations = [
            'Paris', 'London', 'Tokyo', 'Sydney', 'Rome', 'Berlin',
            'Madrid', 'Beijing', 'Moscow', 'Cairo', 'Delhi', 'Bangkok',
            'Seoul', 'Vienna', 'Oslo', 'Lisbon', 'Dublin', 'Prague',
            'Athens', 'Warsaw', 'Budapest', 'Stockholm', 'Copenhagen', 'Brussels'
        ]
        
        self.objects = [
            'book', 'phone', 'laptop', 'pen', 'key', 'wallet',
            'watch', 'camera', 'tablet', 'notebook', 'badge', 'card',
            'bottle', 'bag', 'document', 'file', 'letter', 'package',
            'disk', 'drive', 'mouse', 'keyboard', 'headset', 'charger'
        ]
        
        self.places = [
            'office', 'home', 'library', 'cafe', 'park', 'store',
            'hotel', 'airport', 'station', 'museum', 'theater', 'restaurant',
            'school', 'hospital', 'bank', 'mall', 'gym', 'studio',
            'lab', 'factory', 'warehouse', 'clinic', 'salon', 'bakery'
        ]
        
        self.colors = [
            'red', 'blue', 'green', 'yellow', 'black', 'white',
            'purple', 'orange', 'pink', 'brown', 'gray', 'silver',
            'gold', 'navy', 'teal', 'maroon', 'crimson', 'azure',
            'violet', 'indigo', 'magenta', 'cyan', 'lime', 'coral'
        ]
        
        self.adjectives = [
            'big', 'small', 'tall', 'short', 'old', 'new',
            'fast', 'slow', 'bright', 'dark', 'clean', 'dirty',
            'hot', 'cold', 'soft', 'hard', 'loud', 'quiet',
            'heavy', 'light', 'thick', 'thin', 'wide', 'narrow'
        ]
        
        # Filler words that add length without changing meaning
        self.filler_words = [
            'very', 'quite', 'really', 'actually', 'just', 'even',
            'still', 'also', 'too', 'indeed', 'certainly', 'surely',
            'clearly', 'obviously', 'definitely', 'absolutely', 'totally',
            'completely', 'entirely', 'fully', 'rather', 'somewhat'
        ]
        
        self.connecting_phrases = [
            'in fact', 'as a matter of fact', 'to be honest',
            'to tell the truth', 'by the way', 'speaking of which',
            'come to think of it', 'now that I think about it'
        ]
    
    def generate_qa_pair(self, context_length: str = 'short') -> Tuple[str, str, str, int]:
        """
        Generate a single Q&A pair with specified context length.
        
        Args:
            context_length: 'short', 'medium', or 'long'
            
        Returns:
            (context, question, answer, context_word_count)
        """
        entity = random.choice(self.entities)
        location = random.choice(self.locations)
        obj = random.choice(self.objects)
        place = random.choice(self.places)
        color = random.choice(self.colors)
        adj1 = random.choice(self.adjectives)
        adj2 = random.choice(self.adjectives)
        
        # Question type
        q_type = random.choice(['where_person', 'where_object', 'what_color', 'who_found'])
        
        if context_length == 'short':
            # Short: 8-12 words (very concise)
            if q_type == 'where_person':
                context = f"{entity} lives in {location}"
                question = f"where does {entity} live"
                answer = location
            elif q_type == 'where_object':
                context = f"the {obj} is in the {place}"
                question = f"where is the {obj}"
                answer = f"in the {place}"
            elif q_type == 'what_color':
                context = f"the {obj} is {color}"
                question = f"what color is the {obj}"
                answer = color
            else:  # who_found
                context = f"{entity} found the {obj}"
                question = f"who found the {obj}"
                answer = entity
        
        elif context_length == 'medium':
            # Medium: 25-35 words (moderate filler)
            filler = ' '.join(random.sample(self.filler_words, 6))
            
            if q_type == 'where_person':
                context = f"{entity} who is {adj1} and {filler} lives in {location} and works there"
                question = f"where does {entity} live"
                answer = location
            elif q_type == 'where_object':
                context = f"the {adj1} {obj} that is {filler} is located in the {adj2} {place}"
                question = f"where is the {obj}"
                answer = f"in the {place}"
            elif q_type == 'what_color':
                context = f"the {adj1} {obj} which is {filler} is {color} in color"
                question = f"what color is the {obj}"
                answer = color
            else:  # who_found
                context = f"{entity} who is {adj1} {filler} found the {color} {obj} yesterday"
                question = f"who found the {obj}"
                answer = entity
        
        else:  # long
            # Long: 50-70 words (lots of filler)
            filler1 = ' '.join(random.sample(self.filler_words, 10))
            filler2 = ' '.join(random.sample(self.filler_words, 8))
            connector = random.choice(self.connecting_phrases)
            
            if q_type == 'where_person':
                context = (f"{entity} who is {adj1} and {adj2} and {filler1} "
                          f"{connector} lives in the city of {location} which is "
                          f"{filler2} and works at the {place} there every single day")
                question = f"where does {entity} live"
                answer = location
            elif q_type == 'where_object':
                context = (f"the {adj1} {obj} that was {adj2} and {filler1} "
                          f"{connector} is located in the {adj2} {place} which is "
                          f"{filler2} near the center")
                question = f"where is the {obj}"
                answer = f"in the {place}"
            elif q_type == 'what_color':
                context = (f"the {adj1} {obj} which was {adj2} and {filler1} "
                          f"{connector} is {color} in color and {filler2} "
                          f"sits on the table")
                question = f"what color is the {obj}"
                answer = color
            else:  # who_found
                context = (f"{entity} who is {adj1} and {adj2} and {filler1} "
                          f"{connector} found the {color} {obj} which was "
                          f"{filler2} in the {place} yesterday afternoon")
                question = f"who found the {obj}"
                answer = entity
        
        return context, question, answer, len(context.split())
    
    def generate_dataset(self, 
                        n_short: int = 400,
                        n_medium: int = 400, 
                        n_long: int = 400,
                        shuffle: bool = True) -> Tuple[List[Tuple[str, str, str]], List[int]]:
        """
        Generate complete dataset with balanced context lengths.
        
        Args:
            n_short: Number of short context examples
            n_medium: Number of medium context examples
            n_long: Number of long context examples
            shuffle: Whether to shuffle the dataset
            
        Returns:
            (qa_data, context_lengths) where qa_data is list of (context, question, answer)
        """
        qa_data = []
        context_lengths = []
        
        # Generate short contexts
        for _ in range(n_short):
            ctx, q, ans, ctx_len = self.generate_qa_pair('short')
            qa_data.append((ctx, q, ans))
            context_lengths.append(ctx_len)
        
        # Generate medium contexts
        for _ in range(n_medium):
            ctx, q, ans, ctx_len = self.generate_qa_pair('medium')
            qa_data.append((ctx, q, ans))
            context_lengths.append(ctx_len)
        
        # Generate long contexts
        for _ in range(n_long):
            ctx, q, ans, ctx_len = self.generate_qa_pair('long')
            qa_data.append((ctx, q, ans))
            context_lengths.append(ctx_len)
        
        # Shuffle if requested
        if shuffle:
            combined = list(zip(qa_data, context_lengths))
            random.shuffle(combined)
            qa_data, context_lengths = zip(*combined)
            qa_data = list(qa_data)
            context_lengths = list(context_lengths)
        
        return qa_data, context_lengths


def build_vocabulary(qa_pairs: List[Tuple[str, str, str]]) -> Tuple[dict, dict]:
    """
    Build vocabulary from Q&A pairs.
    
    Args:
        qa_pairs: List of (context, question, answer) tuples
        
    Returns:
        (vocab, idx2word) dictionaries
    """
    vocab = {
        "<PAD>": PAD_TOKEN,
        "<SOS>": SOS_TOKEN,
        "<EOS>": EOS_TOKEN,
        "<UNK>": UNK_TOKEN,
        "<SEP>": SEP_TOKEN
    }
    
    for ctx, q, ans in qa_pairs:
        for word in (ctx + " " + q + " " + ans).lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    idx2word = {i: w for w, i in vocab.items()}
    
    return vocab, idx2word


def encode_text(text: str, vocab: dict) -> List[int]:
    """Encode text to indices."""
    return [vocab.get(w.lower(), UNK_TOKEN) for w in text.split()]


def decode_text(indices: List[int], idx2word: dict) -> str:
    """Decode indices to text."""
    words = [idx2word.get(i, "<UNK>") for i in indices 
             if i not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]]
    return " ".join(words)


if __name__ == "__main__":
    # Test the generator
    generator = SyntheticQAGenerator()
    
    print("Testing Synthetic Q&A Generator")
    print("="*80)
    
    for length in ['short', 'medium', 'long']:
        print(f"\n{length.upper()} Context Example:")
        ctx, q, ans, ctx_len = generator.generate_qa_pair(length)
        print(f"  Context ({ctx_len} words): {ctx}")
        print(f"  Question: {q}")
        print(f"  Answer: {ans}")
    
    # Test dataset generation
    print("\n" + "="*80)
    print("Generating dataset...")
    qa_data, context_lengths = generator.generate_dataset(
        n_short=100, n_medium=100, n_long=100
    )
    
    print(f"\nDataset size: {len(qa_data)}")
    print(f"Context length distribution:")
    print(f"  Short (< 15): {sum(1 for l in context_lengths if l < 15)}")
    print(f"  Medium (15-40): {sum(1 for l in context_lengths if 15 <= l < 45)}")
    print(f"  Long (45+): {sum(1 for l in context_lengths if l >= 45)}")
