"""
data.py - Synthetic Q&A Data Generation (From Lesson 4)

This module generates synthetic question-answer pairs with controlled context lengths.
"""

import random


# Special tokens
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
SEP_TOKEN = 4


class SyntheticQAGenerator:
    """
    Generate synthetic Q&A data with controlled context lengths.
    """
    
    def __init__(self, seed=42):
        random.seed(seed)
        
        # Expanded vocabulary for better variety
        self.entities = [
            "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
            "Iris", "Jack", "Kate", "Leo", "Mary", "Nathan", "Olivia", "Peter",
            "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier"
        ]
        
        self.locations = [
            "Paris", "London", "Tokyo", "Sydney", "Rome", "Berlin", "Madrid", "Beijing",
            "Moscow", "Cairo", "Delhi", "Bangkok", "Seoul", "Vienna", "Oslo", "Lisbon",
            "Dublin", "Prague", "Athens", "Warsaw", "Budapest", "Stockholm", "Copenhagen", "Brussels"
        ]
        
        self.objects = [
            "book", "key", "phone", "wallet", "watch", "laptop", "bag", "pen",
            "notebook", "camera", "umbrella", "glasses", "hat", "shoes", "jacket", "bottle",
            "cup", "plate", "clock", "mirror", "lamp", "chair", "table", "box"
        ]
        
        self.places = [
            "office", "park", "library", "cafe", "store", "museum", "school", "hospital",
            "station", "airport", "hotel", "restaurant", "gym", "theater", "mall", "bank",
            "church", "beach", "garden", "studio", "gallery", "lab", "factory", "warehouse"
        ]
        
        self.colors = [
            "red", "blue", "green", "yellow", "black", "white", "purple", "orange",
            "pink", "brown", "gray", "silver", "gold", "cyan", "magenta", "navy",
            "teal", "lime", "maroon", "olive", "coral", "indigo", "violet", "turquoise"
        ]
        
        self.adjectives = [
            "tall", "short", "big", "small", "old", "new", "happy", "sad",
            "fast", "slow", "hot", "cold", "bright", "dark", "loud", "quiet",
            "clean", "dirty", "soft", "hard", "smooth", "rough", "thick", "thin"
        ]
        
        self.filler_words = [
            "who", "is", "very", "quite", "really", "also", "just", "still",
            "even", "many", "some", "few", "most", "all", "each", "every",
            "single", "only", "again", "often", "always", "never"
        ]
        
        self.connecting_phrases = [
            "and", "but", "or", "so", "because", "which", "that", "when"
        ]
    
    def generate_filler(self, length):
        """Generate filler text of approximately given length."""
        words = []
        while len(' '.join(words).split()) < length:
            words.append(random.choice(self.filler_words))
            if random.random() < 0.3:
                words.append(random.choice(self.connecting_phrases))
        return ' '.join(words[:length])
    
    def generate_qa_pair(self, context_length='short'):
        """
        Generate a single Q&A pair with specified context length.
        
        Args:
            context_length: 'short', 'medium', or 'long'
        
        Returns:
            (context, question, answer) tuple
        """
        entity = random.choice(self.entities)
        location = random.choice(self.locations)
        obj = random.choice(self.objects)
        place = random.choice(self.places)
        color = random.choice(self.colors)
        adj = random.choice(self.adjectives)
        
        question_type = random.choice(['where_person', 'where_object', 'what_color', 'who_found'])
        
        if context_length == 'short':
            # 8-12 words: Very concise, minimal filler
            if question_type == 'where_person':
                context = f"{entity} lives in {location}"
                question = f"where does {entity} live"
                answer = location
            elif question_type == 'where_object':
                context = f"the {obj} is in the {place}"
                question = f"where is the {obj}"
                answer = f"in the {place}"
            elif question_type == 'what_color':
                context = f"the {obj} is {color}"
                question = f"what color is the {obj}"
                answer = color
            else:  # who_found
                context = f"{entity} found the {obj}"
                question = f"who found the {obj}"
                answer = entity
        
        elif context_length == 'medium':
            # 25-35 words: Moderate filler
            filler1 = self.generate_filler(3)
            filler2 = self.generate_filler(3)
            
            if question_type == 'where_person':
                context = f"{entity} {filler1} lives in {location} {filler2} and works there"
                question = f"where does {entity} live"
                answer = location
            elif question_type == 'where_object':
                context = f"the {color} {obj} {filler1} is in the {adj} {place} {filler2}"
                question = f"where is the {obj}"
                answer = f"in the {place}"
            elif question_type == 'what_color':
                context = f"the {obj} {filler1} that {entity} has {filler2} is {color}"
                question = f"what color is the {obj}"
                answer = color
            else:  # who_found
                context = f"{entity} {filler1} found the {color} {obj} in the {place} {filler2}"
                question = f"who found the {obj}"
                answer = entity
        
        else:  # long
            # 50-70 words: Extensive filler
            filler1 = self.generate_filler(8)
            filler2 = self.generate_filler(8)
            filler3 = self.generate_filler(6)
            
            if question_type == 'where_person':
                context = f"{entity} who is {adj} and {filler1} lives in the city of {location} which is {filler2} and works at the {adj} {place} {filler3} every single day"
                question = f"where does {entity} live"
                answer = location
            elif question_type == 'where_object':
                context = f"the {color} {obj} which is {adj} and {filler1} is located in the {adj} {place} that {filler2} near the {adj} {location} {filler3}"
                question = f"where is the {obj}"
                answer = f"in the {place}"
            elif question_type == 'what_color':
                context = f"the {obj} that {entity} who is {adj} and {filler1} has and uses {filler2} every day at the {place} {filler3} is {color}"
                question = f"what color is the {obj}"
                answer = color
            else:  # who_found
                context = f"{entity} who is {adj} and {filler1} found the {color} {obj} in the {adj} {place} {filler2} near {location} {filler3}"
                question = f"who found the {obj}"
                answer = entity
        
        return context, question, answer
    
    def generate_dataset(self, n_short=400, n_medium=400, n_long=400, shuffle=True):
        """
        Generate complete dataset with mixed context lengths.
        
        Returns:
            qa_data: List of (context, question, answer) tuples
            context_lengths: List of actual context lengths in words
        """
        data = []
        lengths = []
        
        # Generate short contexts
        for _ in range(n_short):
            ctx, q, ans = self.generate_qa_pair('short')
            data.append((ctx, q, ans))
            lengths.append(len(ctx.split()))
        
        # Generate medium contexts
        for _ in range(n_medium):
            ctx, q, ans = self.generate_qa_pair('medium')
            data.append((ctx, q, ans))
            lengths.append(len(ctx.split()))
        
        # Generate long contexts
        for _ in range(n_long):
            ctx, q, ans = self.generate_qa_pair('long')
            data.append((ctx, q, ans))
            lengths.append(len(ctx.split()))
        
        # Shuffle if requested
        if shuffle:
            combined = list(zip(data, lengths))
            random.shuffle(combined)
            data, lengths = zip(*combined)
            data = list(data)
            lengths = list(lengths)
        
        return data, lengths


def build_vocabulary(qa_data):
    """
    Build vocabulary from Q&A data.
    
    Returns:
        vocab: Dictionary mapping word to index
        idx2word: Dictionary mapping index to word
    """
    vocab = {
        '<PAD>': PAD_TOKEN,
        '<SOS>': SOS_TOKEN,
        '<EOS>': EOS_TOKEN,
        '<UNK>': UNK_TOKEN,
        '<SEP>': SEP_TOKEN
    }
    
    # Collect all words
    for context, question, answer in qa_data:
        for word in context.split() + question.split() + answer.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    # Create reverse mapping
    idx2word = {idx: word for word, idx in vocab.items()}
    
    return vocab, idx2word


def encode_text(text, vocab):
    """Convert text to list of indices."""
    return [vocab.get(word, UNK_TOKEN) for word in text.split()]


def decode_text(indices, idx2word):
    """Convert list of indices to text."""
    return ' '.join([idx2word.get(idx, '<UNK>') for idx in indices])


if __name__ == "__main__":
    # Test the generator
    print("Testing SyntheticQAGenerator...")
    
    generator = SyntheticQAGenerator()
    
    # Generate examples
    for length in ['short', 'medium', 'long']:
        ctx, q, ans = generator.generate_qa_pair(length)
        print(f"\n{length.upper()} ({len(ctx.split())} words):")
        print(f"  Context:  {ctx}")
        print(f"  Question: {q}")
        print(f"  Answer:   {ans}")
    
    # Generate dataset
    qa_data, lengths = generator.generate_dataset(n_short=10, n_medium=10, n_long=10)
    print(f"\n✓ Generated {len(qa_data)} Q&A pairs")
    print(f"  Length range: {min(lengths)}-{max(lengths)} words")
    
    # Test vocabulary
    vocab, idx2word = build_vocabulary(qa_data)
    print(f"✓ Vocabulary size: {len(vocab)}")
