# Self Attention

#### Initializing Dimensions and Random Matrices
We start by defining the dimensions of our matrices and generating random values for them.
Here:

L is the sequence length, which represents the number of tokens or words in the sequence. # input sequece => My name is Amaan
d_k is the dimensionality of the keys and queries.
d_v is the dimensionality of the values.

```Python
import numpy as np
import math

L, d_k, d_v = 4, 8, 8
q = np.random.randn(L, d_k)
k = np.random.randn(L, d_k)
v = np.random.randn(L, d_v)
```
