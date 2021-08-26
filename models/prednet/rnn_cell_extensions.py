""" Extensions to TF RNN class adapted from red (https://github.com/una-dinosauria/human-motion-prediction)"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell
# The import for LSTMStateTuple changes in TF >= 1.2.0
from pkg_resources import parse_version as pv

from experiments import config

if pv(tf.__version__) >= pv('1.2.0'):
    from tensorflow.contrib.rnn import LSTMStateTuple
else:
    from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMStateTuple
del pv


class LinearSpaceDecoderWrapper(RNNCell):
    """Operator adding a linear decoder to an RNN cell"""

    def __init__(self, cell, output_size):
        """
        Args:
          cell: an RNNCell.
        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell

        print('output_size = {0}'.format(output_size))
        print('state_size = {0}'.format(self._cell.state_size))

        # Tuple if multi-rnn
        if isinstance(self._cell.state_size, tuple):
            # Fine if GRU... just select the last tuple value which will be 1024
            insize = self._cell.state_size[-1]

            # LSTMStateTuple if LSTM
            if isinstance(insize, LSTMStateTuple):
                insize = insize.h

        else:
            # Fine if not multi-rnn
            insize = self._cell.state_size

        self.w_out = tf.get_variable("proj_w_out", [insize, output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
        self.b_out = tf.get_variable("proj_b_out", [output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

        self.linear_output_size = output_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self.linear_output_size

    def __call__(self, inputs, state, scope=None):
        # run the rnn as usual
        output, new_state = self._cell(inputs, state, scope)  # input dim = 37

        # apply dropout
        dropout = tf.nn.dropout(output, rate=0.2, name='dropout')

        # linear layer
        output = tf.matmul(dropout, self.w_out) + self.b_out

        return output, new_state


class ResidualWrapper(RNNCell):
    """Operator adding residual connections (the input is added to the output) to a given cell."""
    def __init__(self, cell):
        """
        Args:
          cell: an RNNCell.
        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        # run the rnn rnn+linear_layer
        linear_output, new_state = self._cell(inputs[:, 3:], state, scope)
        # linear_output => 34 dim qpos velocities vector
        # inputs[:, 3:] => for all batches select (qpos+goal_pos)-root_qpos => (34+3-3) = 34
        data_len = config.MOGAZE_SIZE if config.TRAIN_MOGAZE or config.TEST_MOGAZE else config.HUMAN_SIZE
        # Add the residual connection => qvel(34 dim) + qpos (34 dim) = future qpos
        output = tf.add(linear_output, inputs[:, :data_len])

        return output, new_state
