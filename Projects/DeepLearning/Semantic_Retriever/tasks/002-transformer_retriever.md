# Task: Transformer Retriever

## Goal
The goal of this task is to complete the transformer-based retrieval implementation.


## Context
Files to read: 
file: `src/transformer_retriever.py`


## Requirements
1. class TransformerRetriever: load the a pretrained tansformer model. Read the method `build_index`
2. Complete the Semantic Retrieval method `retrieve` on cosine similarity.
- Compares query embeddings with document embeddings using cosine similarity
- Returns the top-k most similar documents for each query
Note: This method allows the system to “understand” which documents are meaningfully related to a question.



## Constraints
- 

## Deliverables
Ensure you these criteria are met:
  - [ ] The retrieve method compare the query embeddings with document embeddings and return the top most similar document


## Report back
- Any questions or considerations for retrieving the data.