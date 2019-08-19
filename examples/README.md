# fastax examples
This folder contains examples of use of the library, as of now they are quite simple,
but the idea is for them to eventually become more complex and show what Jax and fastax are
truly capable of.

[Xor with MLP](xor.ipynb), this example is based on Colin Raffel's great introductory blog post
[You don't know JAX](https://colinraffel.com/blog/you-don-t-know-jax.html), but using fastax's layers and
utilities instead of coding everything from 0.

[Simple Long Short-Term Memory example](lstm.ipynb), this example is the one created by
Victor Zhou in his amazing blog post [An Introduction to Recurrent Neural Networks for Beginners](https://victorzhou.com/blog/intro-to-rnns/),
but we use fastax's LSTM layer instead of his "homemade" RNN
