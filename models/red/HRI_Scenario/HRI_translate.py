"""
taken from red (https://github.com/una-dinosauria/human-motion-prediction)
Simple code for training an RNN for motion prediction.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from experiments import config
from experiments.config import CKPT_DIR
from models.prednet import data_utils
from models.red import seq2seq_model, RED_data_utils


# Learning
def create_flags():
    # ref: delete all flags: https://stackoverflow.com/questions/49916921/how-to-clear-tf-flags
    try:
        tf.app.flags.DEFINE_float("learning_rate", .005, "Learning rate.")
        tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95,
                                  "Learning rate is multiplied by this much. 1 means no decay.")
        tf.app.flags.DEFINE_integer("learning_rate_step", 10000, "Every this many steps, do decay.")
        tf.app.flags.DEFINE_float("max_gradient_norm", 5, "Clip gradients to this norm.")
        tf.app.flags.DEFINE_integer("batch_size", 30, "Batch size to use during training.")
        tf.app.flags.DEFINE_integer("iterations", config.ITERATIONS, "Iterations to train for.")
        # Architecture
        tf.app.flags.DEFINE_string("seq2seq_architecture", "tied", "Seq2seq architecture to use: [basic, tied].")
        tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
        tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
        tf.app.flags.DEFINE_integer("seq_length_in", 50, "Number of frames to feed into the encoder. 25 fps")
        tf.app.flags.DEFINE_integer("seq_length_out", 25, "Number of frames that the decoder has to predict. 25fps")
        tf.app.flags.DEFINE_boolean("omit_one_hot", True, "Whether to remove one-hot encoding from the data")
        tf.app.flags.DEFINE_boolean("residual_velocities", True,
                                    "Add a residual connection that effectively models velocities")
        # Directories
        tf.app.flags.DEFINE_string("data_dir", config.DATA_DIR, "Data directory")
        # tf.app.flags.DEFINE_string("data_dir", os.path.normpath("./data/qpos_data/qpos"), "Data directory")
        tf.app.flags.DEFINE_string("train_dir", CKPT_DIR, "Training directory.")

        tf.app.flags.DEFINE_string("action", config.ACTION,
                                   "The action to train on. all means all the actions, all_periodic means walking, eating and smoking")
        tf.app.flags.DEFINE_string("loss_to_use", "sampling_based",
                                   "The type of loss to use, supervised or sampling_based")
        tf.app.flags.DEFINE_string("architecture", "red", "The network architecture")
        tf.app.flags.DEFINE_alias("a", "architecture")
        tf.app.flags.DEFINE_string("experiment", "msm_with_goal", "Current experiment name")
        tf.app.flags.DEFINE_alias("e", "experiment")
        tf.app.flags.DEFINE_integer("test_every", 100, "How often to compute error on the test set.")
        tf.app.flags.DEFINE_integer("save_every", 100, "How often to compute error on the test set.")
        tf.app.flags.DEFINE_boolean("test", False, "Set to True for testing.")
        tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
        tf.app.flags.DEFINE_integer("load", config.TRAIN_LOAD, "Try to load a previous checkpoint.")

        tf.app.flags.DEFINE_boolean("avoid_goal", False, "Unused")
        tf.app.flags.DEFINE_alias("ag", "avoid_goal")
    except:
        print("Trying to redefine flags, ignoring redefinition.")
        # import traceback
        # traceback.print_exc()
    flags = tf.app.flags.FLAGS
    return flags


def get_dirs(*_):
    train_dir = config.CKPT_DIR
    summaries_dir = os.path.normpath(os.path.join(train_dir, "log"))  # Directory for TB summaries
    return train_dir, summaries_dir


def create_model(session, actions=None, sampling=False, flags=None):
    """Create translation model and initialize or load parameters in session."""
    if flags is None:
        flags = create_flags()
    train_dir, summaries_dir = get_dirs()
    model = seq2seq_model.Seq2SeqModel(
        flags.seq2seq_architecture,
        flags.seq_length_in if not sampling else 50,
        flags.seq_length_out if not sampling else 100,
        flags.size,  # hidden layer size
        flags.num_layers,
        flags.max_gradient_norm,
        flags.batch_size,
        flags.learning_rate,
        flags.learning_rate_decay_factor,
        summaries_dir,
        flags.loss_to_use if not sampling else "sampling_based",
        not flags.omit_one_hot,
        flags.residual_velocities,
        dtype=tf.float32)

    if flags.load <= 0:
        print("Creating model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        return model

    ckpt = tf.train.get_checkpoint_state(train_dir, latest_filename="checkpoint")
    print("train_dir", train_dir)

    if ckpt and ckpt.model_checkpoint_path:
        # Check if the specific checkpoint exists
        if flags.load > 0:
            if os.path.isfile(os.path.join(train_dir, "checkpoint-{0}.index".format(flags.load))):
                ckpt_name = os.path.normpath(os.path.join(os.path.join(train_dir, "checkpoint-{0}".format(flags.load))))
            else:
                raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(flags.load))
        else:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

        print("Loading model {0}".format(ckpt_name))
        model.saver.restore(session, ckpt_name)
        return model
    else:
        print("Could not find checkpoint. Aborting.")
        raise (ValueError, "Checkpoint {0} does not seem to exist".format(ckpt.model_checkpoint_path))


def train(flags):
    """Train a seq2seq model on human motion"""
    train_dir, summaries_dir = get_dirs()
    actions = define_actions(flags.action)

    number_of_actions = len(actions)

    train_set, test_set, data_mean, data_std = read_qpos_data(flags.seq_length_in, flags.seq_length_out, flags.data_dir,
                                                              not flags.omit_one_hot,
                                                              actions=define_actions(flags.action)
                                                              )

    # Limit TF to take a fraction of the GPU memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    device_count = {"GPU": 0} if flags.use_cpu else {"GPU": 1}
    tf_config = tf.ConfigProto(gpu_options=gpu_options, device_count=device_count)
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:

        # === Create the model ===
        print("Creating %d layers of %d units." % (flags.num_layers, flags.size))

        model = create_model(sess, actions)
        model.train_writer.add_graph(sess.graph)
        print("Model created")

        # === This is the training loop ===
        step_time, loss, val_loss = 0.0, 0.0, 0.0
        current_step = 0 if flags.load <= 0 else flags.load + 1
        previous_losses = []
        data_keys = list(train_set.keys())
        batches = math.ceil(len(data_keys) / flags.batch_size)
        # step_time, loss = 0, 0

        for _ in tqdm(range(flags.iterations)):

            start_time = time.time()
            # shuffle data keys in each iteration
            # random.shuffle(data_keys)
            batch_loss = 0

            for batch in range(int(batches)):
                # === Training step ===
                batch_data = model.get_batch(train_set, data_keys, batch)

                # total_frames = flags.seq_length_in + flags.seq_length_out
                sub_batches = 13  # considering 400 sequences (13*25+75 =400)
                total_sub_batch_loss = 0

                for sub_batch in range(int(sub_batches)):
                    encoder_inputs, decoder_inputs, decoder_outputs = model.get_sub_batch(batch_data, sub_batch)
                    _, sub_batch_loss, loss_summary, lr_summary = model.step(sess, encoder_inputs, decoder_inputs,
                                                                             decoder_outputs[:, :, :34], False)
                    model.train_writer.add_summary(loss_summary, current_step)
                    model.train_writer.add_summary(lr_summary, current_step)
                    total_sub_batch_loss += sub_batch_loss

                batch_loss += total_sub_batch_loss
            step_loss = batch_loss / (sub_batches * batches)

            if current_step % 10 == 0:
                print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss))

            step_time += (time.time() - start_time) / flags.test_every
            loss += step_loss / flags.test_every
            current_step += 1

            # === steplearning rate decay ===
            if current_step % flags.learning_rate_step == 0:
                sess.run(model.learning_rate_decay_op)

            # Once in a while, we save checkpoint, print statistics, and run evals i.e, validation.
            if current_step % flags.test_every == 0:

                # === Validation with randomly chosen seeds ===
                forward_only = True

                test_data_keys = list(test_set.keys())
                test_batches = math.ceil(len(test_data_keys) / flags.batch_size)
                # random.shuffle(test_data_keys)
                test_batch_loss = 0
                test_batch_ms_loss = 0

                for test_batch in range(test_batches):
                    # === Testing step ===
                    test_batch_data = model.get_batch(test_set, test_data_keys, test_batch)

                    test_sub_batches = 13  # considering only 400 sequences (13*25+75 =400)

                    total_test_sub_batch_loss = 0
                    total_ms_loss = 0

                    for sub_batch in range(int(test_sub_batches)):
                        encoder_inputs, decoder_inputs, decoder_outputs = model.get_sub_batch(test_batch_data,
                                                                                              sub_batch)
                        sub_batch_loss, ms_loss, loss_summary = model.step(sess, encoder_inputs, decoder_inputs,
                                                                           decoder_outputs[:, :, :34], forward_only)
                        total_test_sub_batch_loss += sub_batch_loss
                        total_ms_loss += ms_loss
                    test_batch_loss += total_test_sub_batch_loss
                    test_batch_ms_loss += total_ms_loss

                val_loss = test_batch_loss / (test_sub_batches * test_batches)  # Loss book-keeping
                avg_ms_loss = test_batch_ms_loss / (test_sub_batches * test_batches)
                val_summary = tf.Summary(value=[tf.Summary.Value(tag='loss/loss', simple_value=val_loss)])
                model.test_writer.add_summary(val_summary, current_step)

                print()
                print("{0: <16} |".format("milliseconds"), end="")
                for ms in [80, 160, 320, 400, 600, 720, 880, 1000]:
                    print(" {0:5d} |".format(ms), end="")
                print()

                # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
                print("{0: <16} |".format(actions[0]), end="")
                # [1,3,7,9,13,24] => the no of timesteps in output/prediction sequnce (total is 25)
                # for training with parameter "--seq_length_out 25"
                for ms in [2, 4, 8, 10, 15, 18, 22, 25]:
                    if flags.seq_length_out >= ms:
                        print(" {0:.3f} |".format(avg_ms_loss[ms - 1]), end="")
                    else:
                        print("   n/a |", end="")
                print()

                print()
                print("============================\n"
                      "Global step:         %d\n"
                      "Learning rate:       %.4f\n"
                      "Step-time (ms):     %.4f\n"
                      "Train loss avg:      %.4f\n"
                      "--------------------------\n"
                      "Val loss:            %.4f\n"
                      "============================" % (model.global_step.eval(),
                                                        model.learning_rate.eval(), step_time * 1000, loss,
                                                        val_loss))
                print()

                previous_losses.append(loss)

                # Save the model
                if current_step % flags.save_every == 0:
                    print("Saving the model...")
                    start_time = time.time()
                    model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'checkpoint')),
                                     global_step=current_step)
                    print("done in {0:.2f} ms".format((time.time() - start_time) * 1000))

                # Reset global time and loss
                step_time, loss = 0, 0

                sys.stdout.flush()


def define_actions(action):
    """
  Define the list of actions we are using.

  Args
    action: String with the passed action. Could be "all"
  Returns
    actions: List of strings of actions
  Raises
    ValueError if the action is not included in H3.6M
  """

    actions = ["walking", "eating", "smoking", "co-existing", "combined"]

    if action in actions:
        return [action]

    if action == "all":
        return actions

    if action == "all_srnn":
        return ["walking", "eating", "smoking", "discussion"]

    raise (ValueError, "Unrecognized action: %d" % action)


def read_qpos_data(seq_length_in, seq_length_out, data_dir, one_hot, actions):
    """
    Loads data for training/testing and normalizes it.

    Args
      actions: list of strings (actions) to load
      seq_length_in: number of frames to use in the burn-in sequence
      seq_length_out: number of frames to use in the output sequence
      data_dir: directory to load the data from
      one_hot: whether to use one-hot encoding per action
    Returns
      train_set: dictionary with normalized training data
      test_set: dictionary with test data
      data_mean: d-long vector with the mean of the training data
      data_std: d-long vector with the standard dev of the training data
      dim_to_ignore: dimensions that are not used becaused stdev is too small
      dim_to_use: dimensions that we are actually using in the model
    """
    # === Read training data ===
    print("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
        seq_length_in, seq_length_out))

    # train_set, complete_train = RED_data_utils.load_qpos_data(data_dir, actions, one_hot, category='train')
    # test_set, _ = RED_data_utils.load_qpos_data(data_dir, actions, one_hot, category='val')

    path_to_data_set = os.path.join(data_dir, "hri_scenarios.h5")
    train_set, complete_train = data_utils.load_data(path_to_data_set, actions[0], category='train')
    test_set, _ = data_utils.load_data(path_to_data_set, actions[0], category='test')

    # data_mean, data_std = data_utils.normalization_stats(complete_train[:, 3:34])

    # Compute normalization stats
    data_mean, data_std, _, _ = RED_data_utils.normalization_stats(complete_train[:, :34])
    # first 3 pos values not normalised they are just x, y and z values of the humnoid in world frame;
    # it can be any value based on how you position the human
    # also last 3 values not normalised that is only goal pos we are not predicting goal pos

    # Normalize -- subtract mean, divide by stdev # data, data_mean, data_std, one_hot
    train_set = RED_data_utils.normalize_data(train_set, data_mean, data_std, one_hot)
    test_set = RED_data_utils.normalize_data(test_set, data_mean, data_std, one_hot)
    print("done reading data.")

    return train_set, test_set, data_mean, data_std


def test(flags, timestep=0, scenario="co-existing"):
    if timestep != 0:
        flags.load = timestep
    else:
        raise (ValueError, "Must give an iteration to read parameters from")

    actions = define_actions(flags.action)

    # Use the CPU if asked to
    # Use the CPU if asked to
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1, allow_growth=True)
    device_count = {"GPU": 0} if flags.use_cpu else {"GPU": 1}

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count=device_count)) as sess:

        tf.set_random_seed(0)
        random.seed(0)
        np.random.seed(0)

        # === Create the model ===
        print("Creating %d layers of %d units." % (flags.num_layers, flags.size))
        # sampling     = True
        pred_model = create_model(sess, actions, flags=flags)
        print("Model created")

        # Load walking data_mean and std_dev
        data_mean = np.genfromtxt(os.path.join(flags.data_dir, 'combined/norm_stat_red/data_mean.csv'),
                                  delimiter=',')
        data_std = np.genfromtxt(os.path.join(flags.data_dir, 'combined/norm_stat_red/data_std.csv'),
                                 delimiter=',')

        # load real data as ground truth
        from models.prednet.data_utils import load_real_data, load_data
        pred_data, _ = load_data(config.HRI_DATA_PATH, scenario, category="test")
        real_data = load_real_data(config.HRI_DATA_PATH, scenario, category="test")

        normed_pred_data = RED_data_utils.normalize_data(pred_data, data_mean, data_std, not flags.omit_one_hot)

        # Make prediction
        qpos_size = 37  # dim in one time step of qpos data
        real_qpos_size = 35
        encoder_inputs = np.zeros((flags.batch_size, flags.seq_length_in - 1, qpos_size), dtype=float)
        decoder_inputs = np.zeros((flags.batch_size, flags.seq_length_out, qpos_size), dtype=float)
        decoder_outputs = np.zeros((flags.batch_size, flags.seq_length_out, 34), dtype=float)
        real_decoder_outputs = np.zeros((flags.batch_size, flags.seq_length_out, real_qpos_size), dtype=float)

        sub_batches = config.TEST_SUB_BATCH_SIZE
        pred_test_loss = 0
        pred_test_ms_loss = 0
        for sub_batch in range(sub_batches):
            # send 50 data in each loop for prediction
            batch_keys = list(normed_pred_data.keys())

            for i in range(flags.batch_size):
                the_key = batch_keys[i]

                # Select the data around the sampled points
                data_sel = normed_pred_data[the_key][sub_batch: config.SEQ_LENGTH_IN + sub_batch, :]
                # Add the data to each batch i.e, i will vary from 0 to 29 => 30 batches
                encoder_inputs[i, :, 0:qpos_size] = data_sel[0:flags.seq_length_in - 1, :]
                decoder_inputs[i, 0, 0:qpos_size] = data_sel[-1, :]

            for i in range(flags.batch_size):
                the_key = batch_keys[i]

                # Select the data around the sampled points
                data_sel = real_data[the_key][
                           config.SEQ_LENGTH_IN + sub_batch: config.SEQ_LENGTH_IN + sub_batch + config.SEQ_LENGTH_OUT,
                           :]
                # Add the data to each batch i.e, i will vary from 0 to 29 => 30 batches
                real_decoder_outputs[i, :, :] = data_sel

            pred_outputs = pred_model.step(sess, encoder_inputs, decoder_inputs, decoder_outputs, forward_only=True,
                                           pred=True, srnn_seeds=False)
            final_pred_outputs = RED_data_utils.post_process(pred_outputs, data_mean, data_std)
            final_pred_outputs = np.stack(final_pred_outputs, axis=1)  # modify dimension to dimensions to 30x25x35
            # real decoder output dimension is 30x25x35

            # MAE
            pred_test_loss += np.mean(np.absolute(np.subtract(real_decoder_outputs, final_pred_outputs)))
            pred_test_ms_loss += np.mean(np.absolute(np.subtract(real_decoder_outputs, final_pred_outputs)),
                                         axis=(0, 2))

        avg_pred_test_loss = pred_test_loss / sub_batches
        print('total prediction test loss : {}'.format(avg_pred_test_loss))

        avg_pred_test_ms_loss = pred_test_ms_loss / sub_batches

        print()
        print("{0: <16} |".format("milliseconds"), end="")
        for ms in [80, 160, 320, 400, 600, 720, 880, 1000]:
            print(" {0:5d} |".format(ms), end="")
        print()

        # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
        print("{0: <16} |".format('walking'), end="")
        # [1,3,7,9,13,24] => the no of timesteps in output/prediction sequnce (total is 25)
        # for training with parameter "--seq_length_out 25"
        for ms in [2, 4, 8, 10, 15, 18, 22, 25]:
            if flags.seq_length_out >= ms:
                print(" {0:.3f} |".format(avg_pred_test_ms_loss[ms - 1]), end="")
            else:
                print("   n/a |", end="")
        print()

        print()


def main(_):
    flags = create_flags()
    if flags.test:
        test(flags)
    else:
        train(flags)


if __name__ == "__main__":
    tf.app.run(main=main)
