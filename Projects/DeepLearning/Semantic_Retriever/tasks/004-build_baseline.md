# Task: BM25 Retriever

## Goal
The goal is to complete the BM25 Retrieval implementation.


## Context
file: `src/bm25_retriever.py`


## Requirements
In bm25_retriever.py, complete the methods to:

1. Tokenize and preprocess the document corpus (lowercase and split into terms)
2. Build a BM25 keyword index and score documents against each query
3. Return the top-k highest-scoring documents per query


## Constraints
- The implementation should be efficient and scalable for large datasets.
- Ensure that the BM25 scores are correctly calculated and normalized.

## Deliverables
- [ ] **BM25 Retrieval Implementation**: Complete implementation of the BM25 algorithm.
- [ ] **Testing**: Test the retriever with sample queries and documents to verify its functionality.

## Report back
- After completing the task, report back with the following:
  - The BM25 scores for each document-query pair.
  - The top-k most relevant documents for each query.
  - Any questions or considerations for improving the retrieval performance.
