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




```Python
num_heads = 8
head_dim = d_model // num_heads
qkv = qkv.reshape(batch_size, sequence_length, num_heads, 3 * head_dim)
qkv.shape
```
num_heads: This parameter specifies the number of attention heads in the multi-head attention mechanism. Instead of computing a single attention function, the multi-head attention mechanism splits the input into multiple parts (or heads) and computes attention separately for each part. This allows the model to focus on different parts of the input sequence simultaneously, capturing various relationships and patterns.

head_dim: The dimension of each attention head. It is calculated as
head_dim = 512 // 8 = 64

qkv = qkv.reshape(batch_size, sequence_length, num_heads, 3 * head_dim)

This reshapes the qkv tensor to [1, 4, 8, 192], where:

batch_size remains 1.
sequence_length remains 4.
num_heads is 8.
3 * head_dim is 192 (since 3 * 64 = 192).

Final Reshaped Tensor
The tensor qkv now has the shape [1, 4, 8, 192]. This means that for each token in the sequence (4 tokens), there are 8 separate attention heads, and each head has a concatenated Q, K, and V vector of size 192 (which is 3 * 64).

Purpose of the Reshaping
Splitting the Attention Heads: The reshaping operation prepares the qkv tensor to be split into individual attention heads. Each head will independently compute attention scores, allowing the model to capture different types of relationships within the input sequence.

Facilitating Parallel Computation: By organizing the tensor in this manner, the model can efficiently compute the attention scores in parallel for each head, leveraging the power of parallel processing in modern hardware (like GPUs).


###### Output

```Python
torch.Size([1, 4, 8, 192])
```

```Python
qkv = qkv.permute(0, 2, 1, 3)
qkv.shape
```
The code qkv.permute(0, 2, 1, 3) rearranges the dimensions of the tensor to [batch_size, num_heads, sequence_length, 3 * head_dim]

###### Output

```Python
torch.Size([1, 4, 8, 192])
```

```Python
q, k, v = qkv.chunk(3, dim=-1)
q.shape, k.shape, v.shape
```
          
The chunk function splits the tensor into equal-sized chunks along the specified dimension.

In this case:

qkv.chunk(3, dim=-1) splits qkv into 3 chunks along the last dimension (dim=-1).
Each chunk will have the shape:

[batch_size, num_heads, sequence_length, head_dim]
So, for our example, each of q, k, and v will have the shape [1, 8, 4, 64].

```Python
(torch.Size([1, 8, 4, 64]),
 torch.Size([1, 8, 4, 64]),
 torch.Size([1, 8, 4, 64]))
```

