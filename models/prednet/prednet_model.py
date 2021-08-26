""" Code for prednet_run model : adapted from red (https://github.com/una-dinosauria/human-motion-prediction) """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from experiments import config
# from utils.evaluation.prednet_run.config import CKPT_DIR, LOAD
from . import modified_rnn_seq2seq
from . import rnn_cell_extensions  # my extensions of the tf repos


class PredNet(object):
    """prednet_run model for human motion prediction"""

    def __init__(self,
                 architecture,
                 source_seq_len,
                 target_seq_len,
                 rnn_size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 summaries_dir,
                 loss_to_use,
                 residual_velocities=False,
                 dtype=tf.float32):
        """
        Args:
          architecture: [basic, tied] whether to tie the encoder and decoder.
          source_seq_len: lenght of the input sequence.
          target_seq_len: lenght of the target sequence.
          rnn_size: number of units in the rnn.
          num_layers: number of rnns to stack.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          summaries_dir: where to log progress for tensorboard.
          loss_to_use: [supervised, sampling_based]. Whether to use ground truth in
            each timestep to compute the loss after decoding, or to feed back the
            prediction from the previous time-step.
          residual_velocities: whether to use a residual connection that models velocities.
          dtype: the data type to use to store internal variables.
        """

        self.human_size = config.MOGAZE_SIZE if config.TRAIN_MOGAZE or config.TEST_MOGAZE else config.HUMAN_SIZE
        self.goal_size = config.GOAL_SIZE
        self.input_size = self.human_size if config.AVOID_GOAL else self.human_size + self.goal_size
        print("Input size is %d" % self.input_size)
        self.train_writer = tf.summary.FileWriter(os.path.normpath(os.path.join(summaries_dir, 'train')))
        self.val_writer = tf.summary.FileWriter(os.path.normpath(os.path.join(summaries_dir, 'val')))
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        print('rnn_size = {0}'.format(rnn_size))
        cell = tf.contrib.rnn.GRUCell(self.rnn_size)

        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.rnn_size) for _ in range(num_layers)])

        # transform input in suitable format for tf
        with tf.name_scope("inputs"):
            enc_in = tf.placeholder(dtype, shape=[None, source_seq_len - 1, self.input_size], name="enc_in")  # dim 37
            dec_in = tf.placeholder(dtype, shape=[None, target_seq_len, self.input_size], name="dec_in")  # dim 37
            # dec_out is the variable to store the ground truth which will be later used to compute loss
            dec_out = tf.placeholder(dtype, shape=[None, target_seq_len, self.human_size], name="dec_out")  # dim 34

            self.encoder_inputs = enc_in
            self.decoder_inputs = dec_in
            self.decoder_outputs = dec_out

            enc_in = tf.transpose(enc_in, [1, 0, 2])
            dec_in = tf.transpose(dec_in, [1, 0, 2])
            dec_out = tf.transpose(dec_out, [1, 0, 2])

            enc_in = tf.reshape(enc_in, [-1, self.input_size])
            dec_in = tf.reshape(dec_in, [-1, self.input_size])
            dec_out = tf.reshape(dec_out, [-1, self.human_size])

            enc_in = tf.split(enc_in, source_seq_len - 1, axis=0)
            dec_in = tf.split(dec_in, target_seq_len, axis=0)
            dec_out = tf.split(dec_out, target_seq_len, axis=0)

        # add linear decoder layer to GRU
        cell = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell, self.human_size)

        # Finally, wrap everything in a residual layer if we want to model velocities
        if residual_velocities:
            print('residual_velocities = {}'.format(residual_velocities))
            cell = rnn_cell_extensions.ResidualWrapper(cell)

        # for sampling based loss, loop function generates next input as previous output
        lf = None
        if loss_to_use == "sampling_based":
            def lf(prev, decoder_inputs, i):
                # assumption: the goal position is fixed during prediction of 25 seq
                if not config.AVOID_GOAL:
                    prev = tf.concat([prev, decoder_inputs[0][:, self.human_size:]], -1)
                return prev
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

        self.outputs = outputs  # 25 tensors of shape (?,34)

        # defining the weighted loss function specific to PredNet
        with tf.name_scope("loss_angles"):
            position_loss = tf.reduce_mean(tf.square(tf.subtract(
                tf.convert_to_tensor(dec_out)[:, :, :3], tf.convert_to_tensor(outputs)[:, :, :3])))
            orientation_loss = tf.reduce_mean(tf.square(tf.subtract(
                tf.convert_to_tensor(dec_out)[:, :, 3:], tf.convert_to_tensor(outputs)[:, :, 3:])))
            loss_angles = 0.2 * position_loss + 0.8 * orientation_loss

        self.loss = loss_angles
        self.loss_summary = tf.summary.scalar('loss/loss', self.loss)

        # loss at each time step (1 time step = 40ms)
        self.ms_loss = tf.reduce_mean(tf.square(tf.subtract(dec_out, outputs)), axis=[1, 2])

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.gradient_norms = norm
        # update the trainable parameters by SGD optimizer
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Keep track of the learning rate
        self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

        # save the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    def step(self, session, encoder_inputs, decoder_inputs, decoder_outputs, forward_only=False, pred=False):
        """Run a step of the model feeding the given inputs.
        Args
          session: tensorflow session to use.
          encoder_inputs: encoder input vector.
          decoder_inputs: decoder inputs  vector.
          decoder_outputs: decoder outputs vector.
          forward_only: whether to do the backward step (training) or only forward (validation or testing).
          test: True for testing.
        Returns
          gradient norm, loss (mean squared error) and summaries for training.
          loss, ms_loss and loss_summary for validation.
          outputs for testing.
        """
        input_feed = {self.encoder_inputs: encoder_inputs,
                      self.decoder_inputs: decoder_inputs,
                      self.decoder_outputs: decoder_outputs}

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
            # validation step
            output_feed = [self.loss,
                           self.ms_loss,
                           self.loss_summary]
            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]

        else:
            # testing step
            output_feed = [self.outputs]
            outputs = session.run(output_feed, input_feed)
            return outputs[0]

    def get_batch(self, data, data_keys, batch):
        """Get a random batch of data
        Args
          data: input data
          data_keys: keys of input data dictionary
          batch: the batch number
        Returns
          data: the batch data
        """
        batch_keys = data_keys[batch * self.batch_size: batch * self.batch_size + self.batch_size]
        random.shuffle(batch_keys)
        data = {batch_key: data[batch_key][5:, :] for batch_key in batch_keys}  # avoiding first 5 time steps
        return data

    def get_sub_batch(self, data, sub_batch, data_keys=None):
        """Get a sub batch of data
        Args
          data: input batch data
          sub_batch: the sub_batch number
        Returns
          tuple (encoder_inputs, decoder_inputs, decoder_outputs)
        """
        batch_keys = list(data_keys) if config.TRAIN_MOGAZE or config.TEST_MOGAZE else list(data.keys())

        # How many frames in total do we need?
        total_frames = self.source_seq_len + self.target_seq_len

        encoder_inputs = np.zeros((self.batch_size, self.source_seq_len - 1, self.input_size), dtype=float)
        decoder_inputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)
        decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)

        for i in range(self.batch_size):
            the_key = batch_keys[i]
            data_sel = data[the_key][sub_batch * self.target_seq_len: sub_batch * self.target_seq_len + total_frames, :]

            encoder_inputs[i, :, 0:self.input_size] = data_sel[0:self.source_seq_len - 1, :self.input_size]
            decoder_inputs[i, :, 0:self.input_size] = data_sel[
                                                      self.source_seq_len - 1:self.source_seq_len + self.target_seq_len - 1,
                                                      :self.input_size]
            decoder_outputs[i, :, 0:self.input_size] = data_sel[self.source_seq_len:, 0:self.input_size]

        return encoder_inputs, decoder_inputs, decoder_outputs

    @staticmethod
    def load_model(use_cpu=False, ckpt_dir=None, timestep=None, **_):
        if ckpt_dir is None:
            ckpt_dir = config.CKPT_DIR
        if timestep is None:
            timestep = config.TEST_LOAD
        # TODO replace with         pred_model = create_model(sess, config.TEST_LOAD)
        """Create model and initialize or load parameters in session."""
        print("Creating rnn_GRU model %d layers of %d units." % (1, 1024))

        # Note: define two separate tf graphs for RL model and DL motion pred model
        # graph for DL model below
        dl_graph = tf.Graph()
        device_count = {"GPU": 0} if use_cpu else {"GPU": 1}  # Use the CPU if asked to
        tf_config = tf.ConfigProto(device_count=device_count)
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config, graph=dl_graph)

        # add operations to tensorflow dl specific graph
        with dl_graph.as_default():
            model = PredNet(
                architecture='tied',
                source_seq_len=config.SEQ_LENGTH_IN,
                target_seq_len=config.SEQ_LENGTH_OUT,
                rnn_size=1024,  # hidden layer size
                num_layers=1,
                max_gradient_norm=5,
                batch_size=1,
                learning_rate=0.001,
                learning_rate_decay_factor=0.95,
                summaries_dir=config.SUMMARIES_DIR,
                loss_to_use="sampling_based",
                residual_velocities=True,
                dtype=tf.float32)

        ckpt = tf.train.get_checkpoint_state(ckpt_dir, latest_filename="checkpoint")
        print("ckpt_dir", ckpt_dir)

        if ckpt and ckpt.model_checkpoint_path:
            # Check if the specific checkpoint exists
            if timestep > 0:
                if os.path.isfile(os.path.join(ckpt_dir, "checkpoint-{0}.index".format(timestep))):
                    ckpt_name = os.path.normpath(
                        os.path.join(os.path.join(ckpt_dir, "checkpoint-{0}".format(timestep))))
                else:
                    raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(timestep))
            else:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            print("Loading model {0}".format(ckpt_name))
            model.saver.restore(sess, ckpt_name)

        return model, dl_graph, sess
