"""
Word2Vec-based Retrieval Module for Semantic Search
"""
import numpy as np
import warnings
import os
import sys

# Suppress gensim warnings and exceptions at module level
warnings.filterwarnings("ignore", category=UserWarning, module="gensim")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

# Redirect stderr temporarily to suppress C-level exceptions from gensim
from io import StringIO
_original_stderr = sys.stderr

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
from typing import List, Optional, Dict

from utils import logger

# Restore stderr
sys.stderr = _original_stderr


class Word2VecRetriever:
    """
    Word2Vec-based retrieval using static word embeddings.
    Demonstrates traditional semantic retrieval before transformers.
    """
    
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, 
                 sg: int = 0, epochs: int = 10):
        """
        Initialize Word2Vec retriever.
        
        Args:
            vector_size (int): Dimensionality of word vectors
            window (int): Context window size
            min_count (int): Minimum word frequency threshold
            sg (int): Training algorithm (0=CBOW, 1=Skip-gram)
            epochs (int): Number of training epochs
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.model = None
        self.corpus_vectors = None
        self.corpus_texts = None
        
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for Word2Vec training.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: Tokenized and cleaned words
        """
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        words = text.split()
        
        # Remove stop words and short words
        words = [word for word in words 
                if word not in ENGLISH_STOP_WORDS and len(word) > 2]
        
        return words
    
    def _text_to_vector(self, text: str) -> Optional[np.ndarray]:
        """
        Convert text to vector by averaging word vectors.
        
        Args:
            text (str): Input text
            
        Returns:
            np.ndarray: Document vector (or None if no words found)
        """
        words = self._preprocess_text(text)
        word_vectors = []
        
        # YOUR CODE HERE: Extract word vectors from trained model
        # For each word, check if it exists in self.model.wv and add to word_vectors
        for word in words:
            # Check if word exists in vocabulary and add vector
            pass
        
        if word_vectors:
            # YOUR CODE HERE: Average word vectors to get document vector
            # Use np.mean() along the correct axis
            return None
        else:
            # Return zero vector if no words found in vocabulary
            return np.zeros(self.vector_size)
    
    def build_index(self, corpus_texts: List[str]) -> None:
        """
        Build Word2Vec model and corpus index.
        
        Args:
            corpus_texts (List[str]): List of documents to index
        """
        logger.info("Building Word2Vec model...")
        
        # YOUR CODE HERE: Preprocess all documents for training
        # Use self._preprocess_text() on each document in corpus_texts
        processed_docs = None
        
        # YOUR CODE HERE: Train Word2Vec model
        # Create Word2Vec instance with self parameters (vector_size, window, etc.)
        self.model = None
        
        logger.info(f"Word2Vec model trained with vocabulary")
        
        # YOUR CODE HERE: Convert all documents to vectors using trained model
        # Use self._text_to_vector() on each document and store in list
        self.corpus_vectors = []
        
        # YOUR CODE HERE: Convert list to numpy array and store corpus_texts
        self.corpus_vectors = None
        self.corpus_texts = corpus_texts
        
        logger.info(f"Indexed {len(corpus_texts)} documents")
    
    def retrieve(self, queries: List[str], k: int = 10) -> Dict[int, List[int]]:
        """
        Retrieve top-k documents for each query using Word2Vec similarity.
        
        Args:
            queries (List[str]): List of query strings
            k (int): Number of documents to retrieve per query
            
        Returns:
            Dict[int, List[int]]: Dictionary mapping query indices to top-k document indices
        """
        if self.model is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        results = {}
        
        for q_idx, query in enumerate(queries):
            # YOUR CODE HERE: Convert query to vector using _text_to_vector
            query_vector = None
            
            if query_vector is not None:
                # YOUR CODE HERE: Calculate cosine similarities between query and corpus
                # Use cosine_similarity from sklearn.metrics.pairwise
                similarities = None
                
                # YOUR CODE HERE: Get top-k most similar document indices
                # Use np.argsort with reverse order and slice to k
                top_indices = None
                results[q_idx] = top_indices.tolist()
            else:
                # If query has no known words, return random documents
                results[q_idx] = list(range(min(k, len(self.corpus_texts))))
        
        return results
    
    def get_vocabulary_stats(self) -> dict:
        """
        Get statistics about the Word2Vec vocabulary.

        Returns:
            dict: Vocabulary statistics
        """
        if self.model is None:
            return {"error": "Model not trained"}

        return {
            "vocabulary_size": len(self.model.wv),
            "vector_size": self.vector_size,
            "total_words_trained": self.model.corpus_count,
            "most_similar_to_search": self.model.wv.most_similar("search", topn=5) if "search" in self.model.wv else None
        }

    def __del__(self):
        """Clean up Word2Vec model to prevent exception warnings on deletion"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                # Properly clean up the model
                del self.model.wv
                del self.model
        except:
            pass  # Silently ignore any cleanup errors

    @staticmethod
    def optimize_parameters(param_combinations, corpus_texts, query_texts, qrels_dict, evaluator_class):
        """
        Optimize Word2Vec parameters using grid search.

        Args:
            param_combinations (List[dict]): List of parameter combinations to test
            corpus_texts (List[str]): Corpus documents for training
            query_texts (List[str]): Query texts for evaluation
            qrels_dict (dict): Ground truth relevance judgments
            evaluator_class: IRMetrics class for evaluation

        Returns:
            dict: Results containing best configuration, metrics, and optimization log
        """
        from contextlib import redirect_stderr
        from io import StringIO

        print(f"🔧 Starting Word2Vec Parameter Optimization")
        print(f"📋 Testing {len(param_combinations)} configurations...")

        # YOUR CODE HERE: Initialize tracking variables
        # Set up best_config, best_score, best_metrics, and results_log
        best_config = None
        best_score = None  # Initialize with appropriate starting value
        best_metrics = None
        results_log = []

        for i, params in enumerate(param_combinations, 1):
            print(f"\n📋 Configuration {i}/{len(param_combinations)}")
            print(f"   Parameters: {params}")

            try:
                # Suppress stderr during model creation/deletion to hide C-level exceptions
                stderr_suppressor = StringIO()

                with redirect_stderr(stderr_suppressor):
                    # YOUR CODE HERE: Create and train Word2Vec retriever with current params
                    # Initialize retriever with **params and build index with corpus_texts
                    test_retriever = None
                    
                    # YOUR CODE HERE: Run retrieval evaluation
                    # Use retriever to get results for query_texts with k=20
                    test_results = None

                    # YOUR CODE HERE: Evaluate performance using evaluator_class
                    # Call evaluate_retrieval with test_results and qrels_dict
                    test_metrics = None

                    # YOUR CODE HERE: Extract key metrics for comparison
                    # Get MRR and Recall@5 from test_metrics
                    mrr_score = None
                    recall_5 = None

                    # YOUR CODE HERE: Log results for this configuration
                    # Append dict with config, mrr, recall_5, and metrics to results_log
                    results_log.append({})

                    # Explicitly delete to trigger cleanup inside suppression context
                    del test_retriever

                print(f"   📊 MRR: {mrr_score:.4f} | Recall@5: {recall_5:.4f}")

                # YOUR CODE HERE: Update best configuration if current is better
                # Compare mrr_score with best_score and update if improved
                if True:  # Replace with proper condition
                    best_score = None
                    best_config = None
                    best_metrics = None
                    print(f"   ✨ NEW BEST! (MRR improved: {best_score:.4f})")

            except Exception as e:
                print(f"   ❌ Failed: {str(e)[:100]}...")
                continue

        print(f"\n🏆 OPTIMIZATION COMPLETE")
        print(f"=" * 50)

        # YOUR CODE HERE: Return optimization results
        # Return dict with best_config, best_score, best_metrics, and results_log
        return {}


if __name__ == "__main__":
    print("="*60)
    print("🔤 Word2Vec Retriever Test")
    print("="*60)
    
    # Test with sample documents
    sample_docs = [
        "The cat sat on the mat with a hat",
        "Dogs are loyal pets and good friends",
        "Machine learning uses algorithms and data",
        "Natural language processing works with text",
        "Information retrieval finds relevant documents"
    ]
    
    sample_queries = [
        "pets and animals",
        "computer science and algorithms"
    ]
    
    try:
        # Initialize and test
        retriever = Word2VecRetriever(vector_size=50, epochs=20)
        retriever.build_index(sample_docs)
        
        # Show vocabulary stats
        stats = retriever.get_vocabulary_stats()
        print(f"📊 Vocabulary: {stats['vocabulary_size']} words")
        
        # Test retrieval
        results = retriever.retrieve(sample_queries, k=3)
        
        for i, query in enumerate(sample_queries):
            print(f"\n🔍 Query: '{query}'")
            print("Top results:")
            for j, doc_idx in enumerate(results[i][:3], 1):
                print(f"  {j}. {sample_docs[doc_idx]}")
        
        print("\n✅ Word2Vec retriever test successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
