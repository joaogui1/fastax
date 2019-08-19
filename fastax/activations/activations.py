from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import jax.numpy as np
from jax.scipy.special import logsumexp, erf


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x, alpha=1.0):
    return np.clip(alpha * x, a_min=0.0)


def leaky_relu(x, alpha=1.0):
    return np.where(x >= 0, alpha * x, 0.01 * x)


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * np.exp(x) - 1)


def gelu(x):
    return x * erf(x)


def softplus(x):
    return np.logaddexp(x, 0.0)


def softmax(x, axis=-1):
    """Apply softmax to a vector of logits x along an axis"""
    numerator = np.exp(x - x.max(axis, keepdims=True))
    return numerator / numerator.sum(axis, keepdims=True)


def logsoftmax(x, axis=-1):
    """Apply log softmax to a vector of logits x along an axis"""
    return x - logsumexp(x, axis, keepdims=True)


def fastvar(x, axis, keepdims):
    """A fast but less numerically-stable variance calculation than np.var."""
    return (
        np.mean(x ** 2, axis, keepdims=keepdims) - np.mean(x, axis, keepdims=keepdims) ** 2
    )
