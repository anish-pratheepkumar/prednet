
""" Extensions to TF RNN class taken from red (https://github.com/una-dinosauria/human-motion-prediction)"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell

# The import for LSTMStateTuple changes in TF >= 1.2.0
from pkg_resources import parse_version as pv

from experiments import config

if pv(tf.__version__) >= pv('1.2.0'):
  from tensorflow.contrib.rnn import LSTMStateTuple
else:
  from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMStateTuple
del pv

from tensorflow.python.ops import variable_scope as vs

import collections
import math

class ResidualWrapper(RNNCell):
  """Operator adding residual connections to a given cell."""

  def __init__(self, cell):
    """Create a cell with added residual connection.

    Args:
      cell: an RNNCell. The input is added to the output.

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
    """Run the cell and add a residual connection."""

    # Run the rnn as usual (This includes rnn+linear_layer)
    output, new_state = self._cell(inputs, state, scope)
    data_len = config.MOGAZE_SIZE if config.TRAIN_MOGAZE or config.TEST_MOGAZE else config.HUMAN_SIZE
    # Add the residual connection
    output = tf.add(output, inputs[:, :data_len])

    return output, new_state

class LinearSpaceDecoderWrapper(RNNCell):
  """Operator adding a linear encoder to an RNN cell"""
  #linear decoder right ?
  def __init__(self, cell, output_size):
    """Create a cell with with a linear encoder in space.

    Args:
      cell: an RNNCell. The input is passed through a linear layer.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")

    self._cell = cell

    print( 'output_size = {0}'.format(output_size) )
    print( ' state_size = {0}'.format(self._cell.state_size) )

    # Tuple if multi-rnn
    if isinstance(self._cell.state_size,tuple):

      # Fine if GRU... just select the last tuple value which will be 1024
      insize = self._cell.state_size[-1]

      # LSTMStateTuple if LSTM
      if isinstance( insize, LSTMStateTuple ):
        insize = insize.h

    else:
      # Fine if not multi-rnn
      insize = self._cell.state_size

    self.w_out = tf.get_variable("proj_w_out",
        [insize, output_size],
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
    """Use a linear layer and pass the output to the cell."""

    # Run the rnn as usual
    output, new_state = self._cell(inputs, state, scope)

    # Apply the multiplication to everything i.e, passing rnn o/p through a linear layer giving o/p dimension of 54 + len(action)
    output = tf.matmul(output, self.w_out) + self.b_out

    return output, new_state
