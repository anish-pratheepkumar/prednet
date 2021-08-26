""" modified tf.contrib.legacy_seq2seq.tied_rnn_seq2seq of tensorflow for prednet_run"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope


# tied_rnn_seq2seq -> modified for this network

def rnn_decoder(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None):
    """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next * prev is a 2D Tensor of
        shape [batch_size x output_size], * i is an integer, the step number
        (when advanced control is needed), * next is a 2D Tensor of shape
        [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
    with variable_scope.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, decoder_inputs, i)
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)  # inp will be sliced to 3 inside the wrapper rnn cell extension
            outputs.append(output)
            if loop_function is not None:
                prev = output
    return outputs, state


def tied_rnn_seq2seq(encoder_inputs,
                     decoder_inputs,
                     cell,
                     loop_function=None,
                     dtype=dtypes.float32,
                     scope=None):
    """RNN sequence-to-sequence model with tied encoder and decoder parameters.

  This model first runs an RNN to encode encoder_inputs into a state vector, and
  then runs decoder, initialized with the last encoder state, on decoder_inputs.
  Encoder and decoder use the same RNN cell and share parameters.

  Args:
    encoder_inputs: A list of 2D Tensors [batch_size x input_size].
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    cell: tf.compat.v1.nn.rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to i-th output in
      order to generate i+1-th input, and decoder_inputs will be ignored, except
      for the first element ("GO" symbol), see rnn_decoder for details.
    dtype: The dtype of the initial state of the rnn cell (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "tied_rnn_seq2seq".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
    with variable_scope.variable_scope("combined_tied_rnn_seq2seq"):
        scope = scope or "tied_rnn_seq2seq"
        _, enc_state = rnn.static_rnn(
            cell, encoder_inputs, dtype=dtype, scope=scope)
        variable_scope.get_variable_scope().reuse_variables()
        return rnn_decoder(
            decoder_inputs,
            enc_state,
            cell,
            loop_function=loop_function,
            scope=scope)
