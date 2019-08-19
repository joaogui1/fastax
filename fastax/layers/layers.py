# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications copyright (C) 2019 <João Guilherme Madeira Araújo>


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools
import operator as op

import numpy as onp
from six.moves import reduce

from jax import lax
from jax import random
import jax.numpy as np
from ..activations import *
from ..initializers import *

# Each layer constructor function returns an (init_fun, apply_fun) pair, where
#   init_fun: takes an rng key and an input shape and returns an
#     (output_shape, params) pair,
#   apply_fun: takes params, inputs, and an rng key and applies the layer.


def Dense(out_dim, W_init=kaiming_uniform(), b_init=normal()):
    """Layer constructor function for a dense (fully-connected) layer."""
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W = W_init(k1, (input_shape[-1], out_dim))
        b = b_init(k2, (out_dim,), stddev=1. / np.sqrt(out_dim))
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return np.dot(inputs, W) + b

    return init_fun, apply_fun


def GeneralConv(
    dimension_numbers,
    out_chan,
    filter_shape,
    strides=None,
    padding="VALID",
    W_init=None,
    b_init=normal(),
):
    """Layer construction function for a general convolution layer."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    W_init = W_init or kaiming_uniform()

    def init_fun(rng, input_shape):
        filter_shape_iter = iter(filter_shape)
        kernel_shape = [
            out_chan
            if c == "O"
            else input_shape[lhs_spec.index("C")]
            if c == "I"
            else next(filter_shape_iter)
            for c in rhs_spec
        ]
        output_shape = lax.conv_general_shape_tuple(
            input_shape, kernel_shape, strides, padding, dimension_numbers
        )
        bias_shape = [out_chan if c == "C" else 1 for c in out_spec]
        bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))
        k1, k2 = random.split(rng)

        b_std = 1. / np.sqrt(np.prod(kernel_shape[:-1]))
        W, b = W_init(k1, kernel_shape), b_init(k2, bias_shape, stddev=b_std)
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return (
            lax.conv_general_dilated(
                inputs, W, strides, padding, one, one, dimension_numbers
            ) + b
        )

    return init_fun, apply_fun


Conv1D = functools.partial(GeneralConv, ("NHC", "HIO", "NHC"))
Conv = functools.partial(GeneralConv, ("NHWC", "HWIO", "NHWC"))

def DepthwiseConv2D(out_chan,
                    filter_shape,
                    strides=None,
                    padding="VALID",
                    W_init=None,
                    b_init=normal(),
                    ):
    one = (1,) * len(filter_shape)
    strides = strides or one
    W_init = W_init or kaiming_uniform()

    def init_fun(rng, input_shape):
        kernel_shape = (filter_shape[0], filter_shape[1], 1,
                        out_chan * input_shape[3])
        output_shape = lax.conv_general_shape_tuple(
            input_shape, kernel_shape, strides, padding,
            ("NHWC", "HWIO", "NHWC")
        )
        bias_shape = tuple(input_shape[0], out_chan * input_shape[3])
        k1, k2 = random.split(rng)

        b_std = (1. / np.sqrt(np.prod(kernel_shape[:-1])))
        W, b = W_init(k1, kernel_shape), b_init(k2, bias_shape, stddev=b_std)
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return (
            lax.conv_general_dilated(
                inputs, W, strides, padding, one, one, 
                ("NHWC", "HWIO", "NHWC"), feature_group_count=inputs[1]
            ) + b
        )

    return init_fun, apply_fun



def GeneralConvTranspose(
    dimension_numbers,
    out_chan,
    filter_shape,
    strides=None,
    padding="VALID",
    W_init=None,
    b_init=normal(),
):
    """Layer construction function for a general transposed-convolution layer."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    W_init = W_init or kaiming_uniform()

    def init_fun(rng, input_shape):
        filter_shape_iter = iter(filter_shape)
        kernel_shape = [
            out_chan
            if c == "O"
            else input_shape[lhs_spec.index("C")]
            if c == "I"
            else next(filter_shape_iter)
            for c in rhs_spec
        ]
        output_shape = lax.conv_transpose_shape_tuple(
            input_shape, kernel_shape, strides, padding, dimension_numbers
        )
        bias_shape = [out_chan if c == "C" else 1 for c in out_spec]
        bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))
        k1, k2 = random.split(rng)
        b_std = (1. / np.sqrt(np.prod(kernel_shape[:-1])))
        W, b = W_init(k1, kernel_shape), b_init(k2, bias_shape, stddev=b_std)
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return lax.conv_transpose(inputs, W, strides, padding, dimension_numbers) + b

    return init_fun, apply_fun


Conv1DTranspose = functools.partial(GeneralConvTranspose, ("NHC", "HIO", "NHC"))
ConvTranspose = functools.partial(GeneralConvTranspose, ("NHWC", "HWIO", "NHWC"))


def LSTM(out_dim, W_init=glorot_uniform(), b_init=normal()):
    def init_fun(rng, input_shape):
        k1, k2 = random.split(rng)
        cell, hidden = b_init(k1, (out_dim,)), b_init(k2, (out_dim,))

        k1, k2, k3 = random.split(k1, num=3)
        forget_W, forget_U, forget_b = (
            W_init(k1, (*input_shape[:-1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )

        k1, k2, k3 = random.split(k1, num=3)
        in_W, in_U, in_b = (
            W_init(k1, (*input_shape[:-1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )

        k1, k2, k3 = random.split(k1, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (*input_shape[:-1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )

        k1, k2, k3 = random.split(k1, num=3)
        change_W, change_U, change_b = (
            W_init(k1, (*input_shape[:-1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )

        output_shape = input_shape[:-1] + (out_dim,)
        return (
            output_shape,
            (
                (cell, hidden),
                (forget_W, forget_U, forget_b),
                (in_W, in_U, in_b),
                (out_W, out_U, out_b),
                (change_W, change_U, change_b),
            ),
        )

    def apply_fun(params, inputs, **kwargs):
        (cell, hidden), (forget_W, forget_U, forget_b), (in_W, in_U, in_b), (
            out_W,
            out_U,
            out_b,
        ), (change_W, change_U, change_b) = params

        for inp in inputs:
            input_gate = sigmoid(np.dot(inp, in_W) + np.dot(hidden, in_U) + in_b)
            change_gate = np.tanh(
                np.dot(inp, change_W) + np.dot(hidden, change_U) + change_b
            )
            forget_gate = sigmoid(
                np.dot(inp, forget_W) + np.dot(hidden, forget_U) + forget_b
            )

            cell = np.multiply(change_gate, input_gate) + np.multiply(cell, forget_gate)

            output_gate = sigmoid(np.dot(inp, out_W) + np.dot(hidden, out_U) + out_b)
            output = np.multiply(output_gate, np.tanh(cell))
            hidden = output

        return output

    return init_fun, apply_fun


def BatchNorm(
    axis=(0, 1, 2),
    epsilon=1e-5,
    center=True,
    scale=True,
    beta_init=zeros,
    gamma_init=ones,
):
    """Layer construction function for a batch normalization layer."""
    _beta_init = lambda rng, shape: beta_init(rng, shape) if center else ()
    _gamma_init = lambda rng, shape: gamma_init(rng, shape) if scale else ()
    axis = (axis,) if np.isscalar(axis) else axis

    def init_fun(rng, input_shape):
        shape = tuple(d for i, d in enumerate(input_shape) if i not in axis)
        k1, k2 = random.split(rng)
        beta, gamma = _beta_init(k1, shape), _gamma_init(k2, shape)
        return input_shape, (beta, gamma)

    def apply_fun(params, x, **kwargs):
        beta, gamma = params
        # TODO(phawkins): np.expand_dims should accept an axis tuple.
        # (https://github.com/numpy/numpy/issues/12290)
        ed = tuple(None if i in axis else slice(None) for i in range(np.ndim(x)))
        beta = beta[ed]
        gamma = gamma[ed]
        mean, var = np.mean(x, axis, keepdims=True), fastvar(x, axis, keepdims=True)
        z = (x - mean) / np.sqrt(var + epsilon)
        if center and scale:
            return gamma * z + beta
        if center:
            return z + beta
        if scale:
            return gamma * z
        return z

    return init_fun, apply_fun


def elementwise(fun, **fun_kwargs):
    """Layer that applies a scalar function elementwise on its inputs."""
    init_fun = lambda rng, input_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: fun(inputs, **fun_kwargs)
    return init_fun, apply_fun


Tanh = elementwise(np.tanh)
Relu = elementwise(relu)
Exp = elementwise(np.exp)
LogSoftmax = elementwise(logsoftmax, axis=-1)
Softmax = elementwise(softmax, axis=-1)
Softplus = elementwise(softplus)
Sigmoid = elementwise(sigmoid)
Elu = elementwise(elu)
LeakyRelu = elementwise(leaky_relu)


def _pooling_layer(reducer, init_val, rescaler=None):
    def PoolingLayer(window_shape, strides=None, padding="VALID"):
        """Layer construction function for a pooling layer."""
        strides = strides or (1,) * len(window_shape)
        rescale = rescaler(window_shape, strides, padding) if rescaler else None
        dims = (1,) + window_shape + (1,)  # NHWC
        strides = (1,) + strides + (1,)

        def init_fun(rng, input_shape):
            out_shape = lax.reduce_window_shape_tuple(
                input_shape, dims, strides, padding
            )
            return out_shape, ()

        def apply_fun(params, inputs, **kwargs):
            out = lax.reduce_window(inputs, init_val, reducer, dims, strides, padding)
            return rescale(out, inputs) if rescale else out

        return init_fun, apply_fun

    return PoolingLayer


MaxPool = _pooling_layer(lax.max, -np.inf)
SumPool = _pooling_layer(lax.add, 0.0)


def _normalize_by_window_size(dims, strides, padding):
    def rescale(outputs, inputs):
        one = np.ones(inputs.shape[1:-1], dtype=inputs.dtype)
        window_sizes = lax.reduce_window(one, 0.0, lax.add, dims, strides, padding)
        return outputs / window_sizes[..., np.newaxis]

    return rescale


AvgPool = _pooling_layer(lax.add, 0.0, _normalize_by_window_size)


def Flatten():
    """Layer construction function for flattening all but the leading dim."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[0], reduce(op.mul, input_shape[1:], 1)
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return np.reshape(inputs, (inputs.shape[0], -1))

    return init_fun, apply_fun


Flatten = Flatten()


def Identity():
    """Layer construction function for an identity layer."""
    init_fun = lambda rng, input_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: inputs
    return init_fun, apply_fun


Identity = Identity()


def FanOut(num):
    """Layer construction function for a fan-out layer."""
    init_fun = lambda rng, input_shape: ([input_shape] * num, ())
    apply_fun = lambda params, inputs, **kwargs: [inputs] * num
    return init_fun, apply_fun


def FanInSum():
    """Layer construction function for a fan-in sum layer."""
    init_fun = lambda rng, input_shape: (input_shape[0], ())
    apply_fun = lambda params, inputs, **kwargs: sum(inputs)
    return init_fun, apply_fun


FanInSum = FanInSum()


def FanInConcat(axis=-1):
    """Layer construction function for a fan-in concatenation layer."""

    def init_fun(rng, input_shape):
        ax = axis % len(input_shape[0])
        concat_size = sum(shape[ax] for shape in input_shape)
        out_shape = input_shape[0][:ax] + (concat_size,) + input_shape[0][ax + 1 :]
        return out_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return np.concatenate(inputs, axis)

    return init_fun, apply_fun


def Dropout(rate, mode="train"):
    """Layer construction function for a dropout layer, turning off
        a the weights with given rate."""

    def init_fun(rng, input_shape):
        return input_shape, ()

    def apply_fun(params, inputs, **kwargs):
        rng = kwargs.get("rng", None)
        if rng is None:
            msg = (
                "Dropout layer requires apply_fun to be called with a PRNG key "
                "argument. That is, instead of `apply_fun(params, inputs)`, call "
                "it like `apply_fun(params, inputs, key)` where `key` is a "
                "jax.random.PRNGKey value."
            )
            raise ValueError(msg)
        if mode == "train":
            keep = random.bernoulli(rng, 1 - rate, inputs.shape)
            return np.where(keep, inputs / (1. - rate), 0.)
        else:
            return inputs

    return init_fun, apply_fun


# Composing layers via combinators


def serial(*layers):
    """Combinator for composing layers in serial.

  Args:
    *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

  Returns:
    A new layer, meaning an (init_fun, apply_fun) pair, representing the serial
    composition of the given sequence of layers.
  """
    nlayers = len(layers)
    init_funs, apply_funs = zip(*layers)

    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_fun(params, inputs, **kwargs):
        rng = kwargs.pop("rng", None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        for fun, param, rng in zip(apply_funs, params, rngs):
            inputs = fun(param, inputs, rng=rng, **kwargs)
        return inputs

    return init_fun, apply_fun


def parallel(*layers):
    """Combinator for composing layers in parallel.

  The layer resulting from this combinator is often used with the FanOut and
  FanInSum layers.

  Args:
    *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

  Returns:
    A new layer, meaning an (init_fun, apply_fun) pair, representing the
    parallel composition of the given sequence of layers. In particular, the
    returned layer takes a sequence of inputs and returns a sequence of outputs
    with the same length as the argument `layers`.
  """
    nlayers = len(layers)
    init_funs, apply_funs = zip(*layers)

    def init_fun(rng, input_shape):
        rngs = random.split(rng, nlayers)
        return zip(
            *[
                init(rng, shape)
                for init, rng, shape in zip(init_funs, rngs, input_shape)
            ]
        )

    def apply_fun(params, inputs, **kwargs):
        rng = kwargs.pop("rng", None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        return [
            f(p, x, rng=r, **kwargs)
            for f, p, x, r in zip(apply_funs, params, inputs, rngs)
        ]

    return init_fun, apply_fun


def shape_dependent(make_layer):
    """Combinator to delay layer constructor pair until input shapes are known.

  Args:
    make_layer: a one-argument function that takes an input shape as an argument
      (a tuple of positive integers) and returns an (init_fun, apply_fun) pair.

  Returns:
    A new layer, meaning an (init_fun, apply_fun) pair, representing the same
    layer as returned by `make_layer` but with its construction delayed until
    input shapes are known.
  """

    def init_fun(rng, input_shape):
        return make_layer(input_shape)[0](rng, input_shape)

    def apply_fun(params, inputs, **kwargs):
        return make_layer(inputs.shape)[1](params, inputs, **kwargs)

    return init_fun, apply_fun
