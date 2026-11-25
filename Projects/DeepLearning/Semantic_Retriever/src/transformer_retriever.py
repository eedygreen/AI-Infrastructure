"""
Transformer-based Retrieval using Sentence Embeddings
"""
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
from utils import logger


class TransformerRetriever:
    """Semantic retrieval using transformer-based sentence embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model.
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        logger.info(f"🤖 Loading transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.corpus = None
        logger.info(f"✅ Model loaded: {model_name}")
        
    def build_index(self, corpus_texts: List[str]):
        """Build semantic index by encoding corpus texts."""
        logger.info("🧠 Building semantic index...")
        self.corpus = corpus_texts
        
        # Encode corpus using sentence transformer
        self.corpus_embeddings = self.model.encode(corpus_texts)
        
        print(f"✅ Semantic index built for {len(corpus_texts):,} documents")
        if len(corpus_texts) > 0 and self.corpus_embeddings is not None:
            print(f"   Embedding dimension: {self.corpus_embeddings.shape[1]}")
        
    def retrieve(self, query_texts: List[str], k: int = 20) -> Dict[int, List[int]]:
        """Retrieve top-k documents using semantic similarity."""
        if self.corpus_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")
            
        print(f"🔍 Running semantic retrieval for {len(query_texts)} queries...")
        
        # Encode query texts using the transformer model
        query_embeddings = self.model.encode(query_texts)
        
        # Cosine similarity matrix of shape (n_queries, n_docs)
        similarities = cosine_similarity(query_embeddings, self.corpus_embeddings)

        # Indices of the k highest-scoring documents per query, best first
        top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
        results = {q_idx: top_k_indices[q_idx].tolist() for q_idx in range(len(query_texts))}

        print(f"✅ Retrieved top-{k} documents using semantic similarity")
        return results
