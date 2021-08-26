"""
taken from red (https://github.com/una-dinosauria/human-motion-prediction)
Sequence-to-sequence model for human motion prediction.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import variable_scope as vs

import numpy as np
import os
import tensorflow as tf

from experiments import config
from models.prednet import modified_rnn_seq2seq
from models.red import RED_rnn_cell_extensions  # my extensions of the tf repos


class Seq2SeqModel(object):
    """Sequence-to-sequence model for human motion prediction"""

    def __init__(self,
                 architecture,
                 source_seq_len,
                 target_seq_len,
                 rnn_size,  # hidden recurrent layer size
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 summaries_dir,
                 loss_to_use,
                 one_hot=True,
                 residual_velocities=False,
                 dtype=tf.float32):
        """Create the model.

    Args:
      architecture: [basic, tied] whether to tie the decoder and decoder.
      source_seq_len: lenght of the input sequence.
      target_seq_len: lenght of the target sequence.
      rnn_size: number of units in the rnn.
      num_layers: number of rnns to stack.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      summaries_dir: where to log progress for tensorboard.
      loss_to_use: [supervised, sampling_based]. Whether to use ground truth in
        each timestep to compute the loss after decoding, or to feed back the
        prediction from the previous time-step.
      number_of_actions: number of classes we have.
      one_hot: whether to use one_hot encoding during train/test (sup models).
      residual_velocities: whether to use a residual connection that models velocities.
      dtype: the data type to use to store internal variables.
    """

        self.human_size = config.MOGAZE_SIZE if config.TRAIN_MOGAZE or config.TEST_MOGAZE else config.HUMAN_SIZE
        self.goal_size = config.GOAL_SIZE
        self.input_size = self.human_size + self.goal_size

        print("One hot is ", one_hot)
        print("Input size is %d" % self.input_size)

        # Summary writers for train and test runs
        self.train_writer = tf.summary.FileWriter(os.path.normpath(os.path.join(summaries_dir, 'train')))
        self.test_writer = tf.summary.FileWriter(os.path.normpath(os.path.join(summaries_dir, 'test')))

        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # === Create the RNN that will keep the state ===
        print('rnn_size = {0}'.format(rnn_size))
        cell = tf.contrib.rnn.GRUCell(self.rnn_size)

        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.rnn_size) for _ in range(num_layers)])

        # === Transform the inputs ===
        with tf.name_scope("inputs"):
            # enc_in has source_seq_len-1 because the last frame of encoder_inputs is given as a seed to the decoder (1st input to decoder.)
            enc_in = tf.placeholder(dtype, shape=[None, source_seq_len - 1, self.input_size], name="enc_in")
            dec_in = tf.placeholder(dtype, shape=[None, target_seq_len, self.input_size], name="dec_in")
            # dec_out is the variable to store the ground truth which will be later used to compute loss
            dec_out = tf.placeholder(dtype, shape=[None, target_seq_len, self.human_size], name="dec_out")

            self.encoder_inputs = enc_in  # shape (?, 49, 55)
            self.decoder_inputs = dec_in  # shape (?, 25, 55)
            self.decoder_outputs = dec_out  # shape (?, 25, 55)

            enc_in = tf.transpose(enc_in, [1, 0, 2])  # shape (49, ?, 55)
            dec_in = tf.transpose(dec_in, [1, 0, 2])  # shape (25, ?, 55)
            dec_out = tf.transpose(dec_out, [1, 0, 2])  # shape (25, ?, 55)

            enc_in = tf.reshape(enc_in, [-1, self.input_size])  # shape (?, 55)
            dec_in = tf.reshape(dec_in, [-1, self.input_size])  # shape (?, 25)
            dec_out = tf.reshape(dec_out, [-1, self.human_size])  # shape (?, 25))

            enc_in = tf.split(enc_in, source_seq_len - 1, axis=0)  # split enc_in into 49 tensors of dimension (?,55)
            dec_in = tf.split(dec_in, target_seq_len, axis=0)  # split dec_in into 25 vectors of dimension (?,55)
            dec_out = tf.split(dec_out, target_seq_len, axis=0)  # split dec_out into 25 vectors of dimension (?,55)

        # === Add space decoder ===
        cell = RED_rnn_cell_extensions.LinearSpaceDecoderWrapper(cell, self.human_size)

        # Finally, wrap everything in a residual layer if we want to model velocities
        if residual_velocities:
            cell = RED_rnn_cell_extensions.ResidualWrapper(cell)

        # Store the outputs here
        outputs = []

        # Define the loss function ( ??? loop function right?)
        lf = None
        if loss_to_use == "sampling_based":
            def lf(prev, decoder_inputs, i):  # function for sampling_based loss+6
                next_input = tf.concat([prev, decoder_inputs[0][:, self.human_size:]],
                                       -1)  # assumption is that the goal position is fixed during prediction
                return next_input
        elif loss_to_use == "supervised":
            pass
        else:
            raise (ValueError, "unknown loss: %s" % loss_to_use)

        # Build the RNN
        if architecture == "basic":
            # Basic RNN does not have a loop function in its API, so copying here.
            with vs.variable_scope("basic_rnn_seq2seq"):
                _, enc_state = tf.contrib.rnn.static_rnn(cell, enc_in, dtype=tf.float32)  # Encoder
                outputs, self.states = tf.contrib.legacy_seq2seq.rnn_decoder(dec_in, enc_state, cell,
                                                                             loop_function=lf)  # Decoder
        elif architecture == "tied":
            outputs, self.states = modified_rnn_seq2seq.tied_rnn_seq2seq(enc_in, dec_in, cell, loop_function=lf)
        else:
            raise (ValueError, "Uknown architecture: %s" % architecture)

        self.outputs = outputs  # 25 tensors of shape (?,54 + len(actions) if one hot encoded)

        # defining the loss function
        with tf.name_scope("loss_angles"):
            loss_angles = tf.reduce_mean(tf.square(tf.subtract(dec_out, outputs)))

        self.loss = loss_angles
        self.loss_summary = tf.summary.scalar('loss/loss', self.loss)

        self.ms_loss = tf.reduce_mean(tf.square(tf.subtract(dec_out, outputs)),
                                      axis=[1, 2])  # loss at each timestep(1TS = 40ms)

        # Gradients and SGD update operation for training the model.
        # collect all trainable parameters of the network
        params = tf.trainable_variables()
        # define the optimizer for training
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)

        # Update all the trainable parameters
        # compute gradients of trainable params w.r.t the losses
        gradients = tf.gradients(self.loss, params)
        # clip the gradients w.r.t max_gradient_norm parameter
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.gradient_norms = norm
        # update the trainable parameters by SGD optimizer
        self.updates = opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

        # Keep track of the learning rate
        self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=70)

    def step(self, session, encoder_inputs, decoder_inputs, decoder_outputs,
             forward_only, pred=False, srnn_seeds=False):
        """Run a step of the model feeding the given inputs.

        Args
          session: tensorflow session to use.
          encoder_inputs: list of numpy vectors to feed as encoder inputs.
          decoder_inputs: list of numpy vectors to feed as decoder inputs.
          decoder_outputs: list of numpy vectors that are the expected decoder outputs.
          forward_only: whether to do the backward step or only forward.
          srnn_seeds: True if you want to evaluate using the sequences of SRNN
        Returns
          A triple consisting of gradient norm (or None if we did not do backward),
          mean squared error, and the outputs.
        Raises
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        input_feed = {self.encoder_inputs: encoder_inputs,
                      self.decoder_inputs: decoder_inputs,
                      self.decoder_outputs: decoder_outputs}

        # Output feed: depends on whether we do a backward step or not.
        if not srnn_seeds:
            if not forward_only:

                # Training step
                output_feed = [self.updates,  # Update Op that does SGD.
                               self.gradient_norms,  # Gradient norm.
                               self.loss,
                               self.loss_summary,
                               self.learning_rate_summary]

                outputs = session.run(output_feed, input_feed)
                return outputs[1], outputs[2], outputs[3], outputs[4]  # Gradient norm, loss, summaries

            elif forward_only and not pred:
                # Validation step, not on SRNN's seeds
                output_feed = [self.loss,  # Loss for this batch.
                               self.ms_loss,
                               self.loss_summary]

                outputs = session.run(output_feed, input_feed)
                return outputs[0], outputs[1], outputs[2]  # No gradient norm

            else:
                # prediction step
                output_feed = [self.outputs]
                outputs = session.run(output_feed, input_feed)
                return outputs[0]

        else:
            # Validation on SRNN's seeds
            output_feed = [self.loss,  # Loss for this batch.
                           self.outputs,
                           self.loss_summary]

            outputs = session.run(output_feed, input_feed)

            return outputs[0], outputs[1], outputs[2]  # No gradient norm, loss, outputs.

    def get_batch(self, data, shuffled_data_keys, batch):
        """Get a random batch of data from the specified bucket, prepare for step.

Args
  data: a list of sequences of size n-by-d to fit the model to.
  shuffled_data_keys: shuffled keys of data dictionary
  batch: The batch number
Returns
  The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
  the constructed batches have the proper format to call step(...) later.
"""

        # Select entries at random
        # all_keys    = list(data.keys())
        batch_keys = shuffled_data_keys[batch * self.batch_size: batch * self.batch_size + self.batch_size]

        data = {batch_key: data[batch_key][5:, :] for batch_key in batch_keys}  # avoiding first 5 timesteps

        return data

    def get_sub_batch(self, data, sub_batch, data_keys=None):
        batch_keys = list(data_keys) if config.TRAIN_MOGAZE or config.TEST_MOGAZE else list(data.keys())

        # How many frames in total do we need?
        total_frames = self.source_seq_len + self.target_seq_len

        encoder_inputs = np.zeros((self.batch_size, self.source_seq_len - 1, self.input_size), dtype=float)
        decoder_inputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)
        decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)

        for i in range(self.batch_size):
            the_key = batch_keys[i]

            # Select the data around the sampled points
            data_sel = data[the_key][sub_batch * self.target_seq_len: sub_batch * self.target_seq_len + total_frames, :]

            # Add the data to each batch i.e, i will vary from 0 to 29 => 30 batches
            encoder_inputs[i, :, 0:self.input_size] = data_sel[0:self.source_seq_len - 1, :]
            decoder_inputs[i, :, 0:self.input_size] = data_sel[
                                                      self.source_seq_len - 1:self.source_seq_len + self.target_seq_len - 1,
                                                      :]
            decoder_outputs[i, :, 0:self.input_size] = data_sel[self.source_seq_len:, 0:self.input_size]

        return encoder_inputs, decoder_inputs, decoder_outputs
