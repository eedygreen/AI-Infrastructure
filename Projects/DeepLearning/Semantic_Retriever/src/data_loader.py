"""
Data Loading Module for Semantic Retrieval Project using Official BEIR Package
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import pathlib
from utils import logger

# BEIR imports
from beir import util
from beir.datasets.data_loader import GenericDataLoader, HFDataLoader


def load_beir_dataset(dataset_name, split="test"):
    """
    Load BeIR dataset using the Hugging Face.
    
    Args:
        dataset_name (str): Dataset name (e.g., 'nq')
        split (str): 'train', 'dev', or 'test' (default: 'test')
    
    Returns:
        tuple: (corpus, queries, qrels)
            - corpus: dict of {doc_id: {'title': str, 'text': str}}
            - queries: dict of {query_id: str}
            - qrels: dict of {query_id: {doc_id: relevance_score}}
    """
    # Setup paths
    try:
        logger.info(f"Loading {dataset_name} dataset from Hugging Face...")

        out_dir = os.path.join(pathlib.Path.cwd().parent, "dataset")
        os.makedirs(out_dir, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = out_dir

        # Load the data
        logger.info(f"Loading {split} split...")
        corpus, queries, qrels = HFDataLoader(hf_repo=f"BeIR/{dataset_name}").load(split=split)

        return corpus, queries, qrels
    except Exception as e:
        logger.error(f"error loading dataset: {e}")


def load_beir_nq(split="test"):
    """
    Load BeIR/nq (Natural Questions) dataset using the Hugging Face.
    
    Args:
        split (str): 'train', 'dev', or 'test' (default: 'test')
    
    Returns:
        tuple: (corpus, queries, qrels)
            - corpus: dict of {doc_id: {'title': str, 'text': str}}
            - queries: dict of {query_id: str}
            - qrels: dict of {query_id: {doc_id: relevance_score}}
    """
    return load_beir_dataset("nq", split)





class DataLoader:
    """
    Handles loading and preprocessing of datasets for semantic retrieval using official BEIR package.
    """
    
    def __init__(self, dataset_name: str = "BeIR/nq"):
        """
        Initialize DataLoader for BeIR datasets.
        
        Args:
            dataset_name (str): Dataset name ("BeIR/nq" for Natural Questions)
        """
        self.dataset_name = dataset_name
        self.corpus_dict = None
        self.queries_dict = None
        self.qrels_dict = None
        
    def load_dataset(self, split: str = "test", query_sample_size: Optional[int] = 500, 
                    random_seed: int = 42) -> Dict:
        """
        Load dataset using official BEIR package with intelligent sampling.
        Samples queries randomly and includes all their related corpus documents.
        
        Args:
            split (str): Dataset split ('train', 'dev', 'test')
            query_sample_size (int, optional): Number of queries to sample (default: 500)
            random_seed (int): Random seed for reproducible sampling
            
        Returns:
            Dict: Loaded dataset with corpus, queries, qrels
        """
        try:
            if self.dataset_name == "BeIR/nq":
                logger.info(f"Loading BeIR/nq (Natural Questions) dataset using official BEIR package...")
                corpus, queries, qrels = load_beir_nq(split=split)
                
            else:
                raise ValueError(f"Dataset {self.dataset_name} not supported. Supported dataset: 'BeIR/nq'")
            
            # Intelligent sampling: sample queries and include all related documents
            if query_sample_size and len(queries) > query_sample_size:
                logger.info(f"Sampling {query_sample_size} queries and their related documents...")
                
                # Set random seed for reproducibility
                np.random.seed(random_seed)
                
                # Randomly sample query IDs
                all_query_ids = list(queries.keys())
                sampled_query_ids = np.random.choice(all_query_ids, size=query_sample_size, replace=False).tolist()
                
                # Filter queries to sampled ones
                queries = {q_id: queries[q_id] for q_id in sampled_query_ids}
                
                # Find all document IDs needed for sampled queries
                needed_doc_ids = set()
                sampled_qrels = {}
                
                for query_id in sampled_query_ids:
                    if query_id in qrels:
                        sampled_qrels[query_id] = qrels[query_id]
                        # Collect all document IDs for this query
                        needed_doc_ids.update(qrels[query_id].keys())
                
                # Filter corpus to only needed documents
                original_corpus_size = len(corpus)
                corpus = {doc_id: corpus[doc_id] for doc_id in needed_doc_ids if doc_id in corpus}
                qrels = sampled_qrels
                
                logger.info(f"Smart sampling completed:")
                logger.info(f"  - Sampled {len(queries):,} queries from {len(all_query_ids):,}")
                logger.info(f"  - Included {len(corpus):,} related documents from {original_corpus_size:,}")
                logger.info(f"  - Maintained all {len(qrels):,} query-document relationships")
            
            self.corpus_dict = corpus
            self.queries_dict = queries
            self.qrels_dict = qrels
            
            logger.info(f"Dataset loaded successfully:")
            logger.info(f"  - Corpus: {len(corpus):,} documents")
            logger.info(f"  - Queries: {len(queries):,} queries")
            logger.info(f"  - QRELs: {len(qrels):,} query-document pairs")
            
            return {
                'corpus': corpus,
                'queries': queries,
                'qrels': qrels
            }
                
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            logger.error("Make sure you have installed the BEIR package: pip install beir")
            raise
    
    def prepare_retrieval_data(self) -> Tuple[List[str], List[str], Dict]:
        """
        Convert BEIR format to retrieval format with sequential IDs.
        
        Returns:
            Tuple[List[str], List[str], Dict]: corpus_texts, query_texts, qrels_sequential
        """
        if self.corpus_dict is None or self.queries_dict is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        logger.info("Converting BEIR format to retrieval format...")
        
        # Convert corpus to list with sequential indexing
        corpus_texts = []
        doc_id_to_idx = {}
        
        for idx, (doc_id, doc_info) in enumerate(self.corpus_dict.items()):
            corpus_texts.append(doc_info['text'])
            doc_id_to_idx[doc_id] = idx
        
        # Convert queries to list with sequential indexing
        query_texts = []
        query_id_to_idx = {}
        
        for idx, (query_id, query_text) in enumerate(self.queries_dict.items()):
            query_texts.append(query_text)
            query_id_to_idx[query_id] = idx
        
        # Convert qrels to sequential format
        qrels_sequential = {}
        if self.qrels_dict:
            for query_id, relevant_docs in self.qrels_dict.items():
                if query_id in query_id_to_idx:
                    q_idx = query_id_to_idx[query_id]
                    qrels_sequential[q_idx] = {}
                    
                    for doc_id, score in relevant_docs.items():
                        if doc_id in doc_id_to_idx:
                            doc_idx = doc_id_to_idx[doc_id]
                            qrels_sequential[q_idx][doc_idx] = score
        
        logger.info(f"Converted to retrieval format:")
        logger.info(f"  - Corpus texts: {len(corpus_texts)}")
        logger.info(f"  - Query texts: {len(query_texts)}")
        logger.info(f"  - QRELs with sequential IDs: {len(qrels_sequential)}")
        
        return corpus_texts, query_texts, qrels_sequential
