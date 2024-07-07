# Self Attention

##### Summary
Initialization: Generate query, key, and value matrices.

Matrix Multiplication: Calculate Q. K^T

Scaling: Divide by sqrt Dk

Masking: Apply a mask to prevent attending to future tokens.

Softmax: Apply softmax to get attention weights.

Weighted Sum: Compute the final values by multiplying attention weights with the value matrix.




#### Initializing Dimensions and Random Matrices
We start by defining the dimensions of our matrices and generating random values for them.
Here:

L is the sequence length, which represents the number of tokens or words in the sequence. input sequece => My name is Amaan
d_k is the dimensionality of the keys and queries.
d_v is the dimensionality of the values.

```Python
import numpy as np
import math

L, d_k, d_v = 4, 8, 8
```

#### Standard Normal Distribution
The standard normal distribution is a specific type of normal distribution with a mean of 0 and a standard deviation of 1. It is represented by a bell-shaped curve that is symmetric around the mean. In this distribution, about 68% of the data falls within one standard deviation of the mean, 95% within two standard deviations, and 99.7% within three standard deviations.

This distribution is crucial in statistics because it serves as a reference point for standardizing and comparing different datasets. By converting data to a standard normal distribution, statisticians can use common methods and tools to analyze and interpret data, facilitating consistency and accuracy in statistical analysis.

q (queries), k (keys), and v (values) are randomly initialized matrices of shapes (L, d_k) for q and k, and (L, d_v) for v.

###### np.random.randn generates samples from the standard normal distribution (mean 0, variance 1).


```Python
q = np.random.randn(L, d_k)
k = np.random.randn(L, d_k)
v = np.random.randn(L, d_v)
```

### Print the output

```Python
print("Q\n", q)
print("K\n", k)
print("V\n", v)
```

Q

> [[ 0.11672673 -2.54870451 -1.44065948  0.93661829  1.36278968  1.04252277 
   -0.01310938 -1.3163937 ]
 [ 0.26721599 -0.90218255  0.07417847 -0.10430246  0.52684253 -0.07081531 
  -0.60511725 -0.55225527]
 [-0.93297509  0.28724456  1.37184579  0.41589874  0.34981245 -0.24753755 
  -1.24497125  0.05044148]
 [-0.11414585 -0.01545749 -0.58376828 -0.40193907  0.93931836 -1.94334363 
  -0.34770465  1.50103406]]

K
> [[ 1.1226585  -0.85645535  0.54315044  1.36560451  0.52539476 -0.94502504
  -0.48444661  0.46268014]
 [-0.53713766 -1.16937329 -0.57988617  0.92713577 -0.85995607 -0.40352635
   0.26555146 -1.83159914]
 [-2.06994435 -0.09514715 -1.64928361 -0.17375184  0.13146819 -1.76335363
   1.56568846  0.69751826]
 [ 0.32910684 -0.1939204  -0.80444134  0.78816869  0.35599408  0.28309835
  -0.25970963  1.49744622]]

V
> [[-0.00368231  1.43739233 -0.59614565 -1.23171219  1.12030717 -0.98620738
  -0.15461465 -1.03106383]
 [ 0.85585446 -1.79878344  0.67321704  0.05607552 -0.15542661 -1.41264124
  -0.40136933 -1.17626611]
 [ 0.50465335  2.28693419  0.67128338  0.2506863   1.78802234  0.14775751
  -0.11405725  0.88026286]
 [-0.68069105  0.68385101  0.17994557 -1.68013201  0.91543969 -0.19108312
   0.03160471  1.40527326]]
