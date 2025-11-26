Overview of the Attention Block
Attention Block Function: Computes attention scores and updates embedding values using these scores.
Components: The attention block involves several steps, including calculating query, key, and value matrices and updating embedding vectors.
Step-by-Step Implementation
Query Matrix (Q)
Calculation: Multiply the matrix of embedding vectors by another matrix ( W_Q ) to produce the query matrix (Q).
Notation: Represented as Q for simplicity, often visualized as an array of vectors.
Key Matrix (K)
Calculation: Similar to Q, the key matrix (K) is obtained by multiplying the embedding vectors by matrix ( W_K ).
Concept: Queries can be seen as questions about other words, while keys provide answers.
Computing Attention Scores
Similarity Check: Calculate the dot product between vectors in the Q and K matrices to determine the similarity.
Matrix Multiplication: Multiply the Q matrix by the transposed K matrix to get the attention scores.
Normalization with Softmax
Purpose: Ensures all values are between 0 and 1 and sum to one.
Application: Apply the softmax function to each column of the resulting matrix.
Diagram of Operations
Steps Recap:
Derive Q and K matrices from the input embeddings.
Compute attention scores via matrix multiplication and softmax.
Use these scores to update embedding vectors.
Value Matrix (V)
Derivation: Similar to Q and K, the value matrix (V) is derived using another matrix ( W_V ).
Updating Embeddings: Multiply each vector in V by the corresponding attention scores to produce updated embeddings.
Final Diagram of the Attention Block
Complete Process:
Compute Q, K, and V matrices.
Compute attention scores from Q and K.
Update embeddings using attention scores and V.
Multi-Head Attention
Purpose: Multiple attention heads allow the model to learn different relationships between words in a sentence.
Process: Each head performs similar calculations with different parameters, and the results are concatenated to form new embedding vectors.
Formula for Attention
From "Attention Is All You Need" Paper:
Components: Uses Q, K, and V matrices.
Adjustment: Includes a division by the square root of the dimension of the K matrix vector to prevent large attention scores.
Final Steps: Apply softmax and multiply by V to compute the final output.
Understanding the attention mechanism is key to grasping how Transformer models process and generate text, enabling them to focus on relevant parts of the input sequence when making predictions.