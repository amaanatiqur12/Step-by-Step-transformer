# Self Attention

#### Summary
Initialization : Generate query, key, and value matrices.

Matrix Multiplication : Calculate Q. K^T

Scaling Divide by sqrt Dk

Masking : Apply a mask to prevent attending to future tokens.

Softmax : Apply softmax to get attention weights.

Weighted Sum : Compute the final values by multiplying attention weights with the value matrix.




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


##### performs matrix multiplication between the query matrix q and the transpose of the key matrix k.T.

```Python
np.matmul(q, k.T)
```
The operation np.matmul(q, k.T) performs matrix multiplication between the query matrix q and the transpose of the key matrix k.T .
Explanation of np.matmul(q, k.T)
Given:
q is a matrix of shape(L,dk)
k is a matrix of shape(L,dk)
k.T is the transpose of k, which will have shape(dk,L)


The result of np.matmul(q,k.T) will be a matrix of shape (L,L).
Each element in this resulting matrix represents the dot product between a query vector from k.

>array([[ 1.9385252 ,  5.43647918, -0.38370563,  1.24225801],
       [ 1.35187753,  1.19807371, -1.70999851, -0.38129862],
       [ 1.06382646, -0.86860778, -1.86251774, -0.68520405],
       [ 2.21209236, -2.81995366,  5.32327746,  2.24049732]])


#### Why we need sqrt(d_k) in denominator


When computing the attention scores using the dot product of queries (Q) and keys (K), the resulting values can be quite large, especially for large values of dk​
(the dimensionality of the keys and queries). This can lead to very large values in the exponent of the softmax function, causing the softmax to saturate and produce very small gradients, which makes training difficult.

To counteract this, the dot products are scaled by ​sqrt(dk)
This scaling helps to keep the variance of the dot product approximately constant, preventing the softmax function from saturating and making the model easier to train.

```Python
q_var = q.var()
k_var = k.var()
dot_product = np.matmul(q, k.T)
dot_product_var = dot_product.var()

(q_var, k_var, dot_product_var)
```

###### Ouptut

>(0.8672192297664698, 0.9229851723027697, 5.1446872979260165)


#### Variance with Scaling

Now, let's scale the dot product by sqrt(dk):

```Python
scaled = np.matmul(q, k.T) / math.sqrt(d_k)
q.var(), k.var(), scaled.var()
```

###### Output

>(0.8672192297664698, 0.9229851723027697, 0.643085912240752)

With scaling by sqrt(dk), the variance of the scaled dot product (scaled_dot_product.var())
is reduced significantly, making it more comparable to the variances of the queries and keys.

```Python
scaled
```

>array([[ 0.68537216,  1.92208565, -0.13566043,  0.43920453],
       [ 0.47796088,  0.42358302, -0.60457577, -0.13480942],
       [ 0.37611945, -0.30709922, -0.65849946, -0.24225621],
       [ 0.78209275, -0.99700418,  1.88206279,  0.79213542]])


#### Masking

Masking is a technique used in machine learning, specifically in models like Transformers, to control which elements in a sequence can "see" or "attend to" each other. It ensures that certain positions in the sequence are ignored during computations, which is essential for tasks where the order of information matters.

Simple Example:

Imagine you're trying to predict the next word in a sentence. When predicting the third word, you should only consider the first and second words, not the fourth or fifth ones. Masking helps enforce this rule by "hiding" the future words from the model.

```Python
mask = np.tril(np.ones( (L, L) ))
mask
```

>array([[1., 0., 0., 0.],
       [1., 1., 0., 0.],
       [1., 1., 1., 0.],
       [1., 1., 1., 1.]])

```python
mask[mask == 0] = -np.infty
mask[mask == 1] = 0
mask
```
<pre>
<code>
array([[  0., -inf, -inf, -inf],
       [  0.,   0., -inf, -inf],
       [  0.,   0.,   0., -inf],
       [  0.,   0.,   0.,   0.]])
</code>
</pre>

```Python
scaled + mask
```

>array([[ 0.68537216,        -inf,        -inf,        -inf],
       [ 0.47796088,  0.42358302,        -inf,        -inf],
       [ 0.37611945, -0.30709922, -0.65849946,        -inf],
       [ 0.78209275, -0.99700418,  1.88206279,  0.79213542]])


