# Multi Head Attention

#### Initializing the variables and values

```Python
sequence_length = 4
batch_size = 1
input_dim = 512
d_model = 512
```

sequence_length = 4: The length of the input sequence, which means there are 4 tokens in each input sequence.

batch_size = 1: The size of the batch, indicating that the input tensor contains only one sequence of tokens. it's like having one sentence or one piece of text that you are processing at a time.

input_dim = 512: The dimensionality of the input features, indicating that each token in the sequence is represented by a 512-dimensional vector.

d_model = 512: The dimensionality of the model, which is used for the output dimensionality of the multi-head attention mechanism. This means that the output of the attention mechanism will also have 512 dimensions.

```Python
x = torch.randn((batch_size, sequence_length, input_dim))
x.size()
```
Output:

```Python
torch.Size([1, 4, 512])
```

Generates a random tensor x with the specified shape [1, 4, 512]. This tensor represents a single sequence of 4 tokens, where each token is a 512-dimensional vector filled with random values. In simple words, it creates a random dataset for one sequence of data.

#### Linear Transformation for Query, Key, and Value Vectors

```python
qkv_layer = nn.Linear(input_dim , 3 * d_model)
```

It creates a layer that will convert each token in the input sequence into three different types of vectors, which are necessary for the multi-head attention mechanism to work. The output dimension is three times the model dimension because it concatenates the Q, K, and V vectors. The input vector is transformed into three separate vectors (queries, keys, and values) using a linear transformation. This involves matrix multiplication with learned weights and the addition of biases, resulting in distinct query, key, and value vectors that capture different aspects of the input data.

Input Vector (512 dimensions) → Linear Layer → Output Vector (1536 dimensions)

The Output Vector (1536 dimensions) is split into three parts:
Query Vector (512 dimensions)
Key Vector (512 dimensions)
Value Vector (512 dimensions)

```python
qkv = qkv_layer(x)
qkv.shape

```

This transformation generates a single tensor qkv that combines the queries (Q), keys (K), and values (V) for the multi-head attention mechanism. The output tensor qkv has three times the dimension of the model because it concatenates the Q, K, and V vectors for each token in the sequence.

```Python
torch.Size([1, 4, 1536])
```    


1: Represents the batch size. In this case, there is 1 batch.
4: Represents the sequence length or the number of tokens in each sequence. Here, there are 4 tokens.
1536: Represents the total dimensionality of the concatenated Q, K, and V vectors for each token in the sequence. In your setup, each token is represented by a vector of size 1536
