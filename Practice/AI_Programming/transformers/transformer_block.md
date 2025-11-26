Overview of a Transformer Block
Input Processing: Each input token is converted into an embedding vector and passed to the multi-head attention block.
Feedforward Network: The output from the multi-head attention block is sent to a feedforward network layer.
Chaining: A Transformer model consists of multiple such blocks chained together.
Feedforward Layer
Function: Applied to each embedding vector individually.
Process:
Linear Layer: Each embedding vector is passed to a linear layer with an output size four times larger than the input size.
Activation Function: The GELU activation function is applied to introduce nonlinearity.
Size Reduction: The output vector is then reduced to the original embedding size.
GELU Activation Function
Comparison: GELU is similar to the ReLU activation function but has a smoother shape, which helps in achieving better training results.
Additional Components
Residual Connections
Purpose: Helps with the vanishing gradient problem in deep networks.
Implementation: The input to a block is added to the block's output before passing it forward, improving training efficiency.
Layer Normalization and Dropout
Layer Normalization: Scales the output so that the mean becomes zero and variance becomes one, leading to faster training.
Dropout Layer: Helps prevent overfitting by setting a portion of the output to zero during training.
Summary
Components of a Transformer Block:
Multi-Head Attention
Feedforward Network
Layer Normalization
Dropout
Residual Connections