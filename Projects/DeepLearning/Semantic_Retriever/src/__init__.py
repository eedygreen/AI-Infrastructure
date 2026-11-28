"""
Semantic Retrieval Project - Core Modules

This package contains the core modules for building a semantic retrieval system:
- data_loader: Load and preprocess datasets
- embedding_generator: Generate embeddings using transformer models
- retriever: Implement semantic search and retrieval
- evaluator: Evaluate retrieval performance
"""

from src.data_loader import DataLoader
from src.bm25_retriever import BM25Retriever
from src.transformer_retriever import TransformerRetriever
from src.evaluator import IRMetrics
from src.word2vec_retriever import Word2VecRetriever
from src.utils import *

__version__ = "1.0.0"
__author__ = "I.Idris"

__all__ = [
    "DataLoader",
    "BM25Retriever", 
    "TransformerRetriever",
    "IRMetrics",
    "Word2VecRetriever",
    "*",
]
