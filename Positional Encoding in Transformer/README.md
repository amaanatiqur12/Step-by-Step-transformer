# Theory of Positional Encoding of Transformer


In the context of Transformers, positional encoding is used to provide information about the position of tokens in a sequence since the model itself does not inherently understand the order of the tokens.

The sine and cosine functions are used for positional encoding because they provide a way to encode positions with a unique and smooth representation. Here’s why they are particularly useful:

1. Unique and Continuous Representation
Sine and cosine functions provide a unique and continuous way to represent positions. This is crucial because the model needs to differentiate between different positions in the sequence. By encoding positions with these functions, each position gets a unique representation that varies smoothly across different positions.

2. Capturing Relative Positions
The choice of sine and cosine functions allows the model to easily capture the relative distance between positions. Since these functions have different frequencies, they encode positions in a way that makes it straightforward to compute the difference between positions. For example, the difference in positional encodings for positions 𝑝1 and 𝑝2 will reveal their relative distance, which is beneficial for the model’s understanding of the sequence.

3. Avoiding Arbitrary Collisions
Sine and cosine functions have periodic properties, which helps in avoiding collisions where different positions might otherwise be mapped to the same encoding vector. This periodicity ensures that encodings are unique and evenly distributed across the range of positions.

4. Smooth Gradients
These functions provide smooth gradients, which are useful for the model during training. The smoothness helps in maintaining consistency in how positions are encoded and ensures that similar positions have similar encodings, which is beneficial for gradient-based optimization.

5. Scalability to Longer Sequences
The formula for positional encoding uses different frequencies of sine and cosine functions, which allows it to scale to longer sequences. By using a range of frequencies, the encoding can handle varying sequence lengths effectively.

---
 sine and cosine functions are periodic, meaning they repeat after a certain interval. However, in the context of positional encoding for Transformers, this periodicity is managed carefully to avoid issues.

Managing Periodicity
Different Frequencies: The formula for positional encoding uses different frequencies for sine and cosine functions. By using a range of frequencies, it ensures that the positional encodings cover a broad range of positions before repeating. This diversity in frequencies helps to distinguish between positions more effectively and reduces the likelihood of collisions or ambiguities.

Long Sequences: For practical sequences, the range of frequencies is chosen to be sufficiently broad so that even for very long sequences, the encodings remain unique and informative. The range of frequencies is designed to ensure that positions in practical sequences don’t fall into the periodic overlap zone where encodings might become indistinguishable.

Dimensionality: The dimensionality of the positional encoding vector plays a role in managing periodicity. Higher-dimensional encodings help in maintaining uniqueness by spreading positional information across multiple dimensions. This way, even if individual sine and cosine functions are periodic, the combined encoding vector in a high-dimensional space reduces the likelihood of repeating patterns.
