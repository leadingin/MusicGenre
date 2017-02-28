import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import RNNCell, _linear
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh


def random_exp_initializer(minval=0, maxval=None, seed=None,
                           dtype=dtypes.float32):
    '''Returns an initializer that generates tensors with an exponential distribution.
    Args:
      minval: A python scalar or a scalar tensor. Lower bound of the range
        of random values to generate.
      maxval: A python scalar or a scalar tensor. Upper bound of the range
        of random values to generate.  Defaults to 1 for float types.
      seed: A Python integer. Used to create random seeds. See
        [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
        for behavior.
      dtype: The data type.
    Returns:
      An initializer that generates tensors with an exponential distribution.
    '''

    def _initializer(shape, dtype=dtype, partition_info=None):
        return tf.exp(random_ops.random_uniform(shape, minval, maxval, dtype, seed=seed))

    return _initializer


# Register the gradient for the mod operation. tf.mod() does not have a gradient implemented.
@ops.RegisterGradient('Mod')
def _mod_grad(op, grad):
    x, y = op.inputs
    gz = grad
    x_grad = gz
    y_grad = tf.reduce_mean(-(x // y) * gz, reduction_indices=[0], keep_dims=True)[0]
    return x_grad, y_grad



class PGRUCell(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, alpha=0.001, r_on_init=0.05, tau_init=6.):
        self._num_units = num_units
        self.alpha = alpha
        self.r_on_init = r_on_init
        self.tau_init = tau_init

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        with tf.variable_scope(scope or type(self).__name__):  # "PGRUCell"
            (h_pre, h_pre) = state

            i_size = input_size.value - 1  # -1 to extract time
            times = array_ops.slice(inputs, [0, i_size], [-1, 1])
            # ------------- PHASED GRU ------------- #

            tau = vs.get_variable(
                "T", shape=[self._num_units],
                initializer=random_exp_initializer(0, self.tau_init), dtype=dtype)

            r_on = vs.get_variable(
                "R", shape=[self._num_units],
                initializer=init_ops.constant_initializer(self.r_on_init), dtype=dtype)

            s = vs.get_variable(
                "S", shape=[self._num_units],
                initializer=init_ops.random_uniform_initializer(0., tau.initialized_value()), dtype=dtype)
            # for backward compatibility (v < 0.12.0) use the following line instead of the above
            # initializer = init_ops.random_uniform_initializer(0., tau), dtype = dtype)

            tau_broadcast = tf.expand_dims(tau, dim=0)
            r_on_broadcast = tf.expand_dims(r_on, dim=0)
            s_broadcast = tf.expand_dims(s, dim=0)

            r_on_broadcast = tf.abs(r_on_broadcast)
            tau_broadcast = tf.abs(tau_broadcast)
            times = tf.tile(times, [1, self._num_units])

            # calculate kronos gate
            phi = tf.div(tf.mod(tf.mod(times - s_broadcast, tau_broadcast) + tau_broadcast, tau_broadcast),
                         tau_broadcast)
            is_up = tf.less(phi, (r_on_broadcast * 0.5))
            is_down = tf.logical_and(tf.less(phi, r_on_broadcast), tf.logical_not(is_up))

            k = tf.select(is_up, phi / (r_on_broadcast * 0.5),
                          tf.select(is_down, 2. - 2. * (phi / r_on_broadcast), self.alpha * phi))

            # ------------- PHASED GRU ------------- #
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                r, u = array_ops.split(1, 2, _linear([inputs, state],
                                                     2 * self._num_units, True, 1.0))
                r, u = sigmoid(r), sigmoid(u)
            with vs.variable_scope("Candidate"):
                c = self._activation(_linear([inputs, r * state], self._num_units, True))
            new_h_tilde = u * state + (1 - u) * c

            # Apply Khronos gate
            new_h = k * new_h_tilde + (1 - k) * h_pre

        return new_h, new_h
