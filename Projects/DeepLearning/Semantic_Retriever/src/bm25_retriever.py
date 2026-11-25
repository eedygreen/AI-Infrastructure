"""
BM25 Retrieval Implementation for Baseline Comparison
"""
import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class BM25Retriever:
    """Traditional keyword-based retrieval using BM25 algorithm."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 retriever with tunable parameters.
        
        Args:
            k1 (float): Controls term frequency saturation (default: 1.2)
            b (float): Controls document length normalization (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.corpus = None
        
    def build_index(self, corpus_texts: List[str]):
        """Build BM25 index from corpus texts."""
        print("🔧 Building BM25 index...")
        self.corpus = corpus_texts
        
        # Lowercase and whitespace-tokenize each document
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        
        # BM25Okapi with the configured term-saturation and length-normalization params
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        
        print(f"✅ BM25 index built for {len(corpus_texts):,} documents")
        
    def retrieve(self, query_texts: List[str], k: int = 20) -> Dict[int, List[int]]:
        """Retrieve top-k documents for each query."""
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
            
        print(f"🔍 Running BM25 retrieval for {len(query_texts)} queries...")
        results = {}
        
        for q_idx, query in enumerate(tqdm(query_texts, desc="Retrieving")):
            # Same tokenization as the corpus
            tokenized_query = query.lower().split()
            
            # BM25 score of every document for this query
            doc_scores = self.bm25.get_scores(tokenized_query)
            
            # Top-k document indices, highest score first
            top_k_indices = np.argsort(doc_scores)[::-1][:k]
            
            results[q_idx] = top_k_indices.tolist()
            
        print(f"✅ Retrieved top-{k} documents for {len(query_texts)} queries")
        return results
