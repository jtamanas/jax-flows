from flax.linen.module import compact
import jax
import jax.numpy as np
from jax import random
import flax.linen as nn

from typing import Any

Array = Any


class MaskedDense(nn.Dense):
    mask: Array = None
    use_context: bool = False

    @compact
    def __call__(self, inputs: Array, context=None) -> Array:
        """
        Taken from flax.linen.Dense.
        Applies a masked linear transformation to the inputs along the last dimension.

        Args:
        inputs: The nd-array to be transformed.

        Returns:
        The transformed input.
        """
        inputs = np.asarray(inputs, self.dtype)
        if context is not None and self.use_context:
            assert (
                inputs.shape[0] == context.shape[0]
            ), "inputs and context must have the same batch size"
            inputs = np.hstack([inputs, context])

        kernel = self.param(
            "kernel", self.kernel_init, (self.mask.shape[0], self.features)
        )
        kernel = np.asarray(kernel, self.dtype)
        kernel = kernel * self.mask
        y = jax.lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            bias = np.asarray(bias, self.dtype)
            y += np.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


def MADE(transform):
    """An implementation of `MADE: Masked Autoencoder for Distribution Estimation`
    (https://arxiv.org/abs/1502.03509).

    Args:
        transform: maps inputs of dimension ``num_inputs`` to ``2 * num_inputs``

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_dim, context_dim=0, hidden_dim=64, **kwargs):
        params, apply_fun = transform(
            rng, input_dim, context_dim=context_dim, hidden_dim=hidden_dim
        )

        def direct_fun(params, inputs, context=None, **kwargs):
            log_weight, bias = apply_fun(params, inputs, context=context).split(
                2, axis=1
            )
            outputs = (inputs - bias) * np.exp(-log_weight)
            log_det_jacobian = -log_weight.sum(-1)
            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, context=None, **kwargs):
            outputs = np.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                log_weight, bias = apply_fun(params, outputs, context=context).split(
                    2, axis=1
                )
                outputs = jax.ops.index_update(
                    outputs,
                    jax.ops.index[:, i_col],
                    inputs[:, i_col] * np.exp(log_weight[:, i_col]) + bias[:, i_col],
                )
            log_det_jacobian = -log_weight.sum(-1)
            return outputs, log_det_jacobian

        return params, direct_fun, inverse_fun

    return init_fun
