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

