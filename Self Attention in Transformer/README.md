# Self Attention

#### Standard Normal Distribution
The standard normal distribution is a specific type of normal distribution with a mean of 0 and a standard deviation of 1. It is represented by a bell-shaped curve that is symmetric around the mean. In this distribution, about 68% of the data falls within one standard deviation of the mean, 95% within two standard deviations, and 99.7% within three standard deviations.

This distribution is crucial in statistics because it serves as a reference point for standardizing and comparing different datasets. By converting data to a standard normal distribution, statisticians can use common methods and tools to analyze and interpret data, facilitating consistency and accuracy in statistical analysis.

###### np.random.randn generates samples from the standard normal distribution (mean 0, variance 1).

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
q = np.random.randn(L, d_k)
k = np.random.randn(L, d_k)
v = np.random.randn(L, d_v)
```
