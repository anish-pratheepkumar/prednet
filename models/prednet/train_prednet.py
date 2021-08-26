""" Code for training prednet_run: adapted from red (https://github.com/una-dinosauria/human-motion-prediction) """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import tensorflow as tf
from tqdm import tqdm

import utils.data_generation.data_utils
from experiments import config
from models.prednet import data_utils
from models.prednet import prednet_model


def create_model(session, load, sampling=False):
    """ Create prednet_run model and initialize or load parameters in session. """

    model = prednet_model.PredNet(
        "tied",
        config.SEQ_LENGTH_IN,
        config.SEQ_LENGTH_OUT,
        config.SIZE,
        config.NUM_LAYERS,
        config.MAX_GRADIENT_NORM,
        config.BATCH_SIZE,
        config.LEARNING_RATE,
        config.LEARNING_RATE_DECAY_FACTOR,
        config.SUMMARIES_DIR,
        config.LOSS_TO_USE if not sampling else "sampling_based",
        config.RESIDUAL_VELOCITIES,
        dtype=tf.float32)

    if load <= 0:
        print("Creating model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        return model

    ckpt = tf.train.get_checkpoint_state(config.TRAIN_DIR, latest_filename="checkpoint")
    print("TRAIN_DIR", config.TRAIN_DIR)

    if ckpt and ckpt.model_checkpoint_path:
        # Check if the specific checkpoint exists
        if load > 0:
            if os.path.isfile(os.path.join(config.TRAIN_DIR, "checkpoint-{0}.index".format(load))):
                ckpt_name = os.path.normpath(
                    os.path.join(os.path.join(config.TRAIN_DIR, "checkpoint-{0}".format(load))))
            else:
                raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(load))
        else:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

        print("Loading model {0}".format(ckpt_name))
        model.saver.restore(session, ckpt_name)
        return model

    else:
        print("Could not find checkpoint. Aborting.")
        raise (ValueError, "Checkpoint {0} does not seem to exist".format(ckpt.model_checkpoint_path))


def check_action(action):
    """
    check the list of actions we are considering. Raises ValueError if the passed action is not in the 'actions' list
    """
    actions = ["walking", "wiping", "lifting", "co-existing", "co-operating", "noise", "combined", "p1_1", "p1_2",
               "p2_1"]

    if action in actions:
        return action

    raise (ValueError, "Unrecognized action: %d" % action)


def read_qpos_data(action, seq_length_in, seq_length_out, data_dir):
    """
    Loads data for training/validating and normalizes it.
    Args
      action: action or scenario to load
      seq_length_in: number of frames to use in the input sequence
      seq_length_out: number of frames to use in the output sequence
      data_dir: directory to load the data from
    Returns
      train_set: dictionary with normalized training data
      val_set: dictionary with normalized val data
      data_mean: 31-long vector with the mean of the training data
      data_std: 31-long vector with the standard dev of the training data
    """
    print("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(seq_length_in, seq_length_out))
    path_to_data_set = os.path.join(data_dir, "hri_scenarios.h5")
    train_set, complete_train = data_utils.load_data(path_to_data_set, action, category='train')
    val_set, _ = data_utils.load_data(path_to_data_set, action, category='test')

    data_mean, data_std = data_utils.normalization_stats(complete_train[:, 3:34])

    # save data_mean and data_std
    # filename_mean = os.path.join(config.NORM_STAT_DIR, 'data_mean.csv')
    # np.savetxt(filename_mean, data_mean, delimiter=',')
    # filename_std = os.path.join(config.NORM_STAT_DIR, 'data_std.csv')
    # np.savetxt(filename_std, data_std, delimiter=',')

    train_set = data_utils.normalize_data(train_set, data_mean, data_std)
    val_set = data_utils.normalize_data(val_set, data_mean, data_std)
    print("done reading data.")

    return train_set, val_set, data_mean, data_std


def read_mogaze_data(seq_length_in, seq_length_out, data_dir):
    """
    Loads data for training/testing and normalizes it. Note that we do not handle calibration of gaze
    as mentioned by authors of Mogaze.

    Args
      seq_length_in: number of frames to use in the burn-in sequence
      seq_length_out: number of frames to use in the output sequence
      data_dir: directory to load the data from
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

    train_set, complete_train = utils.data_generation.data_utils.load_mogaze_data(config.MOGAZE_DATA_DIR,
                                                                                  actions=['p1_1'], limit=53250)
    test_set, _ = utils.data_generation.data_utils.load_mogaze_data(config.MOGAZE_DATA_DIR, actions=['p1_2'],
                                                                    limit=18750)

    # Compute normalization stats
    data_mean, data_std = data_utils.normalization_stats(complete_train[:, 3:66])  # exclude root pos

    # first 3 pos values not normalised they are just x, y and z values of the humnoid in world frame;
    # it can be any value based on how you position the human
    # also last 3 values not normalised that is only goal pos we are not predicting goal pos

    # save data_mean and data_std
    # filename_mean = os.path.join(config.MOGAZE_NORM_STAT_DIR, 'data_mean.csv')
    # np.savetxt(filename_mean, data_mean, delimiter=',')
    # filename_std = os.path.join(config.MOGAZE_NORM_STAT_DIR, 'data_std.csv')
    # np.savetxt(filename_std, data_std, delimiter=',')

    # Normalize -- subtract mean, divide by stdev
    train_set = data_utils.normalize_data(train_set, data_mean, data_std)
    test_set = data_utils.normalize_data(test_set, data_mean, data_std)
    print("done reading data.")

    return train_set, test_set, data_mean, data_std


def print_results(ms_loss, model=None, step_time=None, loss=None, val_loss=None):
    """ Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms """
    print()
    pred_time_steps = [80, 160, 320, 400, 600, 720, 880, 1000]
    if config.TEST_MOGAZE:
        pred_time_steps = [83.33, 166.66, 333.33, 416.66, 625.00, 750.00, 916.66, 1041.66]
    elif config.TRAIN_MOGAZE:
        pred_time_steps = [16.5, 33, 66.5, 83.3, 125, 150, 183, 208]

    print("{0: <16} |".format("milliseconds"), end="")
    for ms in pred_time_steps:
        print(" {0:.2f} |".format(ms), end="")
    print()
    # 1 time step = 40 ms for HRI scenario data, for Mogaze data 1 time step = 8.33ms
    print("{0: <16} |".format('walking'), end="")
    for ms in [2, 4, 8, 10, 15, 18, 22, 25]:
        if config.SEQ_LENGTH_OUT >= ms:
            print(" {0:.3f} |".format(ms_loss[ms - 1]), end="")
        else:
            print("   n/a |", end="")
    print()

    print()
    if model is not None:
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


def train():
    """ train prednet_run on specific action/scenario """

    action = check_action(config.ACTION)

    train_set, val_set, data_mean, data_std = read_qpos_data(
        action, config.SEQ_LENGTH_IN, config.SEQ_LENGTH_OUT, config.DATA_DIR)

    # Limit TF to take a fraction of the GPU memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1, allow_growth=True)
    device_count = {"GPU": 0} if config.USE_CPU else {"GPU": 1}

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count=device_count)) as sess:

        # === Create the model ===
        print("Creating %d layers of %d units." % (config.NUM_LAYERS, config.SIZE))
        model = create_model(sess, config.TRAIN_LOAD)
        model.train_writer.add_graph(sess.graph)
        print("Model created")

        # initialize global training variables
        step_time, loss, val_loss = 0.0, 0.0, 0.0
        current_step = 0 if config.TRAIN_LOAD <= 0 else config.TRAIN_LOAD + 1
        # scenarios = [1, 2, 3]  # 1-> co-existing, 2-> co-operating, 3-> Noise
        data_keys = list(train_set.keys())
        num_batches = math.ceil(len(data_keys) / config.BATCH_SIZE)
        # input data is organized with BATCH_SIZE number of episodes from each scenario, pls check file names in
        # 'combined' data folder. (due to variation in episode length of each scenario)
        # loop_control_list = int(num_batches / len(scenarios)) * scenarios

        for _ in tqdm(range(config.ITERATIONS)):
            current_step += 1
            start_time = time.time()

            # local variables
            # batch = -1
            batch_loss = 0

            # batch loop
            for batch in range(num_batches):
                # batch += 1
                batch_data = model.get_batch(train_set, data_keys, batch)

                sub_batches = config.SUB_BATCH_SIZE
                total_sub_batch_loss = 0

                # sub_batch loop
                for sub_batch in range(int(sub_batches)):
                    encoder_inputs, decoder_inputs, decoder_outputs = model.get_sub_batch(batch_data, sub_batch)
                    _, sub_batch_loss, loss_summary, lr_summary = model.step(sess, encoder_inputs, decoder_inputs,
                                                                             decoder_outputs[:, :, :34], False)
                    model.train_writer.add_summary(loss_summary, current_step)
                    model.train_writer.add_summary(lr_summary, current_step)
                    total_sub_batch_loss += sub_batch_loss
                batch_loss += total_sub_batch_loss
            step_loss = batch_loss / (config.SUB_BATCH_SIZE * num_batches)

            if current_step == 1 or current_step % 10 == 0:
                print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss))

            step_time += (time.time() - start_time) / config.VAL_EVERY
            loss += step_loss / config.VAL_EVERY

            if current_step % config.LEARNING_RATE_STEP == 0:
                sess.run(model.learning_rate_decay_op)

            # perform validation and print statistics
            if current_step % config.VAL_EVERY == 0:

                forward_only = True

                val_data_keys = list(val_set.keys())
                num_val_batches = math.ceil(len(val_data_keys) / config.BATCH_SIZE)
                # val_loop_control_list = int(num_val_batches / len(scenarios)) * scenarios

                # val_batch = -1
                val_batch_loss = 0
                batch_ms_loss = 0

                for val_batch in range(num_val_batches):
                    # val_batch += 1
                    val_batch_data = model.get_batch(val_set, val_data_keys, val_batch)

                    val_sub_batches = config.SUB_BATCH_SIZE
                    total_val_sub_batch_loss = 0
                    total_ms_loss = 0

                    for sub_batch in range(int(val_sub_batches)):
                        encoder_inputs, decoder_inputs, decoder_outputs = model.get_sub_batch(val_batch_data,
                                                                                              sub_batch)
                        val_sub_batch_loss, ms_loss, loss_summary = model.step(sess, encoder_inputs, decoder_inputs,
                                                                               decoder_outputs[:, :, :34],
                                                                               forward_only)
                        total_val_sub_batch_loss += val_sub_batch_loss
                        total_ms_loss += ms_loss

                    val_batch_loss += total_val_sub_batch_loss
                    batch_ms_loss += total_ms_loss

                val_loss = val_batch_loss / (config.SUB_BATCH_SIZE * num_val_batches)
                avg_ms_loss = batch_ms_loss / (config.SUB_BATCH_SIZE * num_val_batches)

                val_summary = tf.Summary(value=[tf.Summary.Value(tag='loss/loss', simple_value=val_loss)])
                model.val_writer.add_summary(val_summary, current_step)
                print_results(avg_ms_loss, model, step_time, loss, val_loss)

                # save checkpoint
                if current_step % config.SAVE_EVERY == 0:
                    print("Saving the model...")
                    start_time = time.time()
                    model.saver.save(sess, os.path.normpath(os.path.join(config.TRAIN_DIR, 'checkpoint')),
                                     global_step=current_step)
                    print("done in {0:.2f} ms".format((time.time() - start_time) * 1000))

                # Reset global time and loss
                step_time, loss = 0, 0

                sys.stdout.flush()


def train_mogaze():
    """Train a seq2seq model on human motion"""

    config.TRAIN_MOGAZE = True

    check_action(config.ACTION)

    # We dont handle calibration as mentioned by Mogaze authors
    train_set, test_set, data_mean, data_std = read_mogaze_data(
        config.SEQ_LENGTH_IN, config.SEQ_LENGTH_OUT, config.MOGAZE_DATA_DIR)

    # Limit TF to take a fraction of the GPU memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1, allow_growth=True)
    device_count = {"GPU": 0} if config.USE_CPU else {"GPU": 1}

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count=device_count)) as sess:

        # === Create the model ===
        print("Creating %d layers of %d units." % (config.NUM_LAYERS, config.SIZE))
        model = create_model(sess, config.TRAIN_LOAD)
        model.train_writer.add_graph(sess.graph)
        print("Model created")

        # initialize global training variables
        step_time, loss, val_loss = 0.0, 0.0, 0.0
        current_step = 0 if config.TRAIN_LOAD <= 0 else config.TRAIN_LOAD + 1

        data_keys = list(train_set.keys())

        for _ in tqdm(range(config.ITERATIONS)):
            start_time = time.time()

            # shuffle data keys in each iteration
            random.shuffle(data_keys)

            # local variables
            sub_batches = config.MOGAZE_SUB_BATCH_SIZE
            total_sub_batch_loss = 0

            for sub_batch in range(int(sub_batches)):
                encoder_inputs, decoder_inputs, decoder_outputs = model.get_sub_batch(train_set, sub_batch,
                                                                                      data_keys=data_keys)
                _, sub_batch_loss, loss_summary, lr_summary = model.step(sess, encoder_inputs, decoder_inputs,
                                                                         decoder_outputs[:, :, :66], False)
                model.train_writer.add_summary(loss_summary, current_step)
                model.train_writer.add_summary(lr_summary, current_step)
                total_sub_batch_loss += sub_batch_loss

            step_loss = total_sub_batch_loss / int(sub_batches)

            if current_step % 10 == 0:
                print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss))

            step_time += (time.time() - start_time) / config.VAL_EVERY
            loss += step_loss / config.VAL_EVERY
            current_step += 1

            # === steplearning rate decay ===
            if current_step % config.LEARNING_RATE_STEP == 0:
                sess.run(model.learning_rate_decay_op)

            # Once in a while, we save checkpoint, print statistics, and run evals i.e, validation.
            if current_step % config.VAL_EVERY == 0:
                forward_only = True

                test_data_keys = list(test_set.keys())
                random.shuffle(test_data_keys)

                test_sub_batches = config.MOGAZE_VAL_SUB_BATCH_SIZE

                total_test_sub_batch_loss = 0
                total_ms_loss = 0

                for sub_batch in range(int(test_sub_batches)):
                    encoder_inputs, decoder_inputs, decoder_outputs = model.get_sub_batch(test_set, sub_batch,
                                                                                          data_keys=test_data_keys)
                    sub_batch_loss, ms_loss, loss_summary = model.step(sess, encoder_inputs, decoder_inputs,
                                                                       decoder_outputs[:, :, :66], forward_only)
                    model.train_writer.add_summary(loss_summary, current_step)
                    model.train_writer.add_summary(lr_summary, current_step)
                    total_test_sub_batch_loss += sub_batch_loss
                    total_ms_loss += ms_loss

                val_loss = total_test_sub_batch_loss / int(test_sub_batches)  # Loss book-keeping
                avg_ms_loss = total_ms_loss / int(test_sub_batches)
                model.val_writer.add_summary(loss_summary, current_step)
                print('assuming 1 frame = 8.33ms')

                print_results(avg_ms_loss, model, step_time, loss, val_loss)

                # save checkpoint
                if current_step % config.SAVE_EVERY == 0:
                    print("Saving the model...")
                    start_time = time.time()
                    model.saver.save(sess, os.path.normpath(os.path.join(config.TRAIN_DIR, 'checkpoint')),
                                     global_step=current_step)
                    print("done in {0:.2f} ms".format((time.time() - start_time) * 1000))

                # Reset global time and loss
                step_time, loss = 0, 0

                sys.stdout.flush()


def run():
    if config.TRAIN_MOGAZE:
        train_mogaze()
    else:
        train()


if __name__ == "__main__":
    run()
