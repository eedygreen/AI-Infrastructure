"""
data.py - Synthetic Comment Data Generation for Content Moderation

This module generates synthetic comments for demonstrating Transformer attention patterns.
Dataset mimics Jigsaw Toxic Comment Classification with various comment types and lengths.
"""

import random
import json


class SyntheticCommentGenerator:
    """
    Generate synthetic comments for content moderation demonstration.
    
    Creates a diverse dataset with:
    - Safe/helpful comments
    - Toxic/offensive comments
    - Comments with ambiguous words (e.g., "bank")
    - Various lengths and complexity
    """
    
    def __init__(self, seed=42):
        random.seed(seed)
        
        # Safe, helpful comment templates
        self.safe_templates = [
            "Thank you for the helpful {adjective} feedback on my {noun}.",
            "I really appreciate your {adjective} contribution to this discussion.",
            "This is a {adjective} explanation of {topic}. Very useful!",
            "Great point about {topic}. I learned something new today.",
            "Your {adjective} perspective on {topic} is really insightful.",
            "I agree with your analysis of {topic}. Well articulated!",
            "Thanks for sharing this {adjective} resource about {topic}.",
            "This {adjective} comment really helped me understand {topic} better.",
            "Excellent question about {topic}. Let me try to explain.",
            "I found your explanation of {topic} very {adjective} and clear.",
        ]
        
        # Toxic comment templates (for detection demonstration)
        self.toxic_templates = [
            "This is stupid and you are an idiot for posting this garbage.",
            "What a waste of time. You clearly have no idea what you are talking about.",
            "Shut up and stop posting this nonsense. Nobody cares about your opinion.",
            "This is the dumbest thing I have ever read. Completely worthless content.",
            "You are wrong and too stupid to understand why. Just give up already.",
            "This trash belongs in the garbage. Delete this immediately.",
            "What kind of moron would believe this obvious nonsense?",
            "Your opinion is worthless and you should feel bad for posting it.",
            "This is a complete joke and so are you. Pathetic attempt.",
            "Stop wasting everyone's time with your idiotic comments and wrong ideas.",
        ]
        
        # Ambiguous word examples (for contextualized embeddings)
        self.ambiguous_templates = [
            "The bank can refuse to lend money to the person by the river bank.",
            "The bat flew out of the cave while the player grabbed his baseball bat.",
            "She works at a plant that manufactures equipment to water the plant.",
            "The bass guitar player caught a large bass fish on his day off.",
            "I need to book a flight and also finish reading my book tonight.",
            "The pitcher threw to the base while I poured water from the pitcher.",
            "The dove flew away peacefully while we dove into the swimming pool.",
            "Can you close the window? The store is close to my house.",
            "I saw a saw in the workshop that could saw through thick wood.",
            "The bear walked by bare trees in the forest during winter.",
        ]
        
        # Long context examples (for attention visualization)
        self.long_context_templates = [
            "In the financial district, the bank on the corner of Main Street can refuse to lend money to individuals who do not meet their strict credit requirements, while further down near the scenic park, visitors enjoy walking along the peaceful river bank where children often play and families have picnics on sunny weekend afternoons.",
            "The research team discovered that the machine learning model with multiple attention heads and deep transformer layers could process natural language much more effectively than traditional recurrent neural networks, especially when dealing with long sequences that contain complex dependencies and require understanding of subtle contextual relationships between distant words in the input text.",
            "When the content moderation system analyzes user comments, it must carefully consider not just individual words but also their relationships, context, and the overall sentiment expressed throughout the entire message, taking into account factors like sarcasm, cultural references, and implicit meanings that might not be immediately obvious from surface-level analysis alone.",
        ]
        
        # Vocabulary for template filling
        self.adjectives = [
            "helpful", "detailed", "thorough", "clear", "insightful",
            "thoughtful", "constructive", "valuable", "excellent", "useful",
            "informative", "comprehensive", "well-written", "articulate", "precise"
        ]
        
        self.nouns = [
            "project", "work", "post", "article", "comment",
            "analysis", "research", "paper", "presentation", "code",
            "design", "implementation", "approach", "solution", "idea"
        ]
        
        self.topics = [
            "machine learning", "natural language processing", "transformers",
            "attention mechanisms", "neural networks", "deep learning",
            "computer vision", "reinforcement learning", "data science",
            "artificial intelligence", "model architecture", "training techniques",
            "optimization algorithms", "transfer learning", "fine-tuning"
        ]
    
    def generate_safe_comment(self):
        """Generate a safe, helpful comment."""
        template = random.choice(self.safe_templates)
        return template.format(
            adjective=random.choice(self.adjectives),
            noun=random.choice(self.nouns),
            topic=random.choice(self.topics)
        )
    
    def generate_toxic_comment(self):
        """Generate a toxic comment (for detection demonstration)."""
        return random.choice(self.toxic_templates)
    
    def generate_ambiguous_comment(self):
        """Generate comment with ambiguous words (same word, different meanings)."""
        return random.choice(self.ambiguous_templates)
    
    def generate_long_context_comment(self):
        """Generate long comment for attention visualization."""
        return random.choice(self.long_context_templates)
    
    def generate_dataset(self, n_safe=2000, n_toxic=500, n_ambiguous=200, 
                        n_long=100, shuffle=True):
        """
        Generate complete dataset with various comment types.
        
        Args:
            n_safe: Number of safe/helpful comments
            n_toxic: Number of toxic comments
            n_ambiguous: Number of ambiguous word comments
            n_long: Number of long context comments
            shuffle: Whether to shuffle the final dataset
        
        Returns:
            comments: List of (text, label, type) tuples
        """
        comments = []
        
        # Generate safe comments
        for _ in range(n_safe):
            text = self.generate_safe_comment()
            comments.append({
                'text': text,
                'label': 'safe',
                'type': 'safe',
                'length': len(text.split())
            })
        
        # Generate toxic comments
        for _ in range(n_toxic):
            text = self.generate_toxic_comment()
            comments.append({
                'text': text,
                'label': 'toxic',
                'type': 'toxic',
                'length': len(text.split())
            })
        
        # Generate ambiguous comments
        for _ in range(n_ambiguous):
            text = self.generate_ambiguous_comment()
            comments.append({
                'text': text,
                'label': 'safe',
                'type': 'ambiguous',
                'length': len(text.split())
            })
        
        # Generate long context comments
        for _ in range(n_long):
            text = self.generate_long_context_comment()
            comments.append({
                'text': text,
                'label': 'safe',
                'type': 'long',
                'length': len(text.split())
            })
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(comments)
        
        return comments
    
    def get_demo_examples(self):
        """
        Get curated examples for demo purposes.
        
        Returns dictionary with specific examples for different demonstrations.
        """
        return {
            'contextualized_embedding': "The bank can refuse to lend money to the person by the river bank.",
            'toxic_example': "This is stupid and you are an idiot for posting this garbage.",
            'safe_example': "Thank you for the helpful and detailed feedback on my project.",
            'long_context': "In the financial district, the bank on the corner of Main Street can refuse to lend money to individuals who do not meet their strict credit requirements, while further down near the scenic park, visitors enjoy walking along the peaceful river bank where children often play and families have picnics on sunny weekend afternoons.",
            'syntax_example': "The model attention mechanism learns to focus on relevant contextual information.",
            'semantic_example': "Machine learning and artificial intelligence are transforming natural language processing."
        }


def save_dataset(comments, filepath):
    """Save dataset to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(comments, f, indent=2)


def load_dataset(filepath):
    """Load dataset from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_statistics(comments):
    """Get dataset statistics."""
    stats = {
        'total': len(comments),
        'safe': sum(1 for c in comments if c['label'] == 'safe'),
        'toxic': sum(1 for c in comments if c['label'] == 'toxic'),
        'avg_length': sum(c['length'] for c in comments) / len(comments),
        'min_length': min(c['length'] for c in comments),
        'max_length': max(c['length'] for c in comments),
    }
    
    # Count by type
    types = {}
    for c in comments:
        types[c['type']] = types.get(c['type'], 0) + 1
    stats['types'] = types
    
    return stats


if __name__ == "__main__":
    # Test the generator
    print("Testing SyntheticCommentGenerator...")
    print("=" * 80)
    
    generator = SyntheticCommentGenerator()
    
    # Generate examples
    print("\nSafe Comment Examples:")
    for i in range(3):
        print(f"  {i+1}. {generator.generate_safe_comment()}")
    
    print("\nToxic Comment Examples:")
    for i in range(3):
        print(f"  {i+1}. {generator.generate_toxic_comment()}")
    
    print("\nAmbiguous Word Examples:")
    for i in range(3):
        print(f"  {i+1}. {generator.generate_ambiguous_comment()}")
    
    # Generate dataset
    print("\n" + "=" * 80)
    print("Generating Full Dataset...")
    comments = generator.generate_dataset(
        n_safe=2000,
        n_toxic=500,
        n_ambiguous=200,
        n_long=100
    )
    
    # Get statistics
    stats = get_statistics(comments)
    print(f"\n✓ Generated {stats['total']} comments")
    print(f"\nDataset Statistics:")
    print(f"  Safe comments:  {stats['safe']}")
    print(f"  Toxic comments: {stats['toxic']}")
    print(f"  Average length: {stats['avg_length']:.1f} words")
    print(f"  Length range:   {stats['min_length']}-{stats['max_length']} words")
    print(f"\nBy Type:")
    for type_name, count in stats['types'].items():
        print(f"  {type_name}: {count}")
    
    print("\n✓ Data generation complete!")
