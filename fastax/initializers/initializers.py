from jax import random
import jax.numpy as np

def _get_fans(shape):
    receptive_field = np.prod(shape[:-2])
    if len(shape) >= 2:
        fan_in, fan_out = shape[-2], shape[-1]
    elif len(shape) == 1:
        fan_in, fan_out = shape[0]
    else:
        fan_in, fan_out = 1.
    fan_in *= receptive_field
    fan_out *= receptive_field
    return fan_in, fan_out

def zeros(key, shape, dtype=np.float32): np.zeros(shape, dtype)
def ones(key, shape, dtype=np.float32): np.ones(shape, dtype)

def uniform(mean=0.):
    def init(key, shape, lim=1., dtype=np.float32):
        return random.uniform(key, shape, dtype, minval=-lim, maxval=lim)
    return init

def normal(mean=0.):
    def init(key, shape, stddev=1., dtype=np.float32):
        return random.normal(key, shape, dtype) * stddev + mean
    return init


def variance_scaling(scale, mode, distribution):
    if scale <= 0.:
        raise ValueError(f"scale must be positive float, {scale} given")
    if mode not in {"fan_in", "fan_out", "fan_avg"}:
        raise ValueError(f"Invalid mode argument: {mode}, must be either fan_in, fan_out or fan_avg")

    def init(rng, shape, dtype=np.float32):
        fan_in, fan_out = _get_fans(shape)
        gain = scale
        if mode == "fan_in":
            gain /= fan_in
        elif mode == "fan_out":
            gain /= fan_out
        elif mode == "fan_avg":
            gain /= (fan_in + fan_out) / 2
        if distribution == "truncated_normal":
            # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = np.sqrt(gain) / .87962566103423978
            return random.truncated_normal(rng, -2, 2, shape, dtype) * stddev
        elif distribution == "normal":
            return random.normal(rng, shape, dtype) * np.sqrt(gain)
        elif distribution == "uniform":
            lim = np.sqrt(3. * gain)
            return random.uniform(rng, shape, dtype, minval=-lim, maxval=lim)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")
    return init

def glorot_uniform(scale=1.):
    return variance_scaling(scale, "fan_avg", "uniform")


def glorot_normal(scale=1.):
    return variance_scaling(scale, "fan_avg", "truncated_normal")


def leccun_uniform(scale=1.):
    return variance_scaling(scale, "fan_in", "uniform")


def leccun_normal(scale=1.):
    return variance_scaling(scale, "fan_in", "truncated_normal")


def kaiming_normal(param=0.):
    return variance_scaling(2.0 / np.sqrt(1 + param**2), "fan_in", "truncated_normal")


def kaiming_uniform(param=0.):
    return variance_scaling(2.0 / np.sqrt(1 + param**2), "fan_in", "uniform")


def orthogonal(scale=1.):
    """Initializer that generates an orthogonal matrix.
    If the shape of the tensor to initialize is two-dimensional, it is initialized
    with an orthogonal matrix obtained from the QR decomposition of a matrix of
    random numbers drawn from a normal distribution.
    If the matrix has fewer rows than columns then the output will have orthogonal
    rows. Otherwise, the output will have orthogonal columns.
    If the shape of the tensor to initialize is more than two-dimensional,
    a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
    is initialized, where `n` is the length of the shape vector.
    The matrix is subsequently reshaped to give a tensor of the desired shape.
    Args:
        scale: multiplicative factor to apply to the orthogonal matrix
    References:
        [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)
        ([pdf](https://arxiv.org/pdf/1312.6120.pdf))
    """
    def init(rng, shape, dtype=np.float32):
        if len(shape) < 2:
            raise ValueError("The tensor to initialize must be at least two-dimensional")
        num_rows = np.prod(shape[:-1])
        num_cols = shape[-1]
        flat_shape = (max(num_rows, num_cols), max(num_rows, num_cols))

        random_mat = random.normal(rng, flat_shape, dtype)
        q, r = np.linalg.qr(random_mat)
        d = np.diag(r)
        q *= np.sign(d)
        if num_rows < num_cols:
            q = np.transpose(q)
        return scale * np.reshape(q, shape)
    return init
