import copy
import os
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import utils.data_generation.data_utils
from experiments import config
from models.prednet import data_utils
from models.prednet.train_prednet import check_action, create_model, print_results


def test(scenario="noise"):
    """
    tests the pretrained prednet_run model and provides the results in MAE.

    inputs specific to testing->
    config.ITERATIONS: iterations for which the model was trained for
    config.TEST_LOAD: iteration at which the model has to be loaded
    config.TEST_DATA: action/scenario specific test data
    """
    if config.TEST_LOAD <= 0:
        raise (ValueError, "Must give an iteration to load parameters from")

    action = check_action(config.ACTION)

    # Use the CPU if asked to
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1, allow_growth=True)
    device_count = {"GPU": 0} if config.USE_CPU else {"GPU": 1}

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count=device_count)) as sess:
        # for reproducibility
        tf.set_random_seed(0)
        random.seed(0)
        np.random.seed(0)

        # Load data_mean and std_dev
        data_mean = np.genfromtxt(os.path.join(config.NORM_STAT_DIR, 'data_mean.csv'), delimiter=',')
        data_std = np.genfromtxt(os.path.join(config.NORM_STAT_DIR, 'data_std.csv'), delimiter=',')

        # ground truth (exclude goal pos)

        # prediction data (include goal pos)
        pred_data, _ = data_utils.load_data(config.HRI_DATA_PATH, scenario, category="test")
        real_data = data_utils.load_real_data(config.HRI_DATA_PATH, scenario, category="test")
        normed_pred_data = data_utils.normalize_data(pred_data, data_mean, data_std)

        # Create the model
        print("Creating %d layers of %d units." % (config.NUM_LAYERS, config.SIZE))
        pred_model = create_model(sess, config.TEST_LOAD)
        print("Model created")

        # Make prediction
        input_size = config.HUMAN_SIZE if config.AVOID_GOAL else config.HUMAN_SIZE + config.GOAL_SIZE
        real_qpos_size = config.OUTPUT_QPOS_SIZE

        encoder_inputs = np.zeros((config.BATCH_SIZE, config.SEQ_LENGTH_IN - 1, input_size), dtype=float)
        decoder_inputs = np.zeros((config.BATCH_SIZE, config.SEQ_LENGTH_OUT, input_size), dtype=float)
        decoder_outputs = np.zeros((config.BATCH_SIZE, config.SEQ_LENGTH_OUT, input_size), dtype=float)
        real_decoder_outputs = np.zeros((config.BATCH_SIZE, config.SEQ_LENGTH_OUT, real_qpos_size), dtype=float)

        sub_batches = config.TEST_SUB_BATCH_SIZE  # 5 co-operating, 6 co-existing, 4 noise
        pred_test_loss = 0
        pred_test_ms_loss = 0

        for sub_batch in tqdm(range(sub_batches)):
            # send 50 data in each loop/sub_batch for prediction
            batch_keys = list(normed_pred_data.keys())

            for i in range(config.BATCH_SIZE):
                # Select the data around the sampled points
                the_key = batch_keys[i]
                data_sel = normed_pred_data[the_key][sub_batch: config.SEQ_LENGTH_IN + sub_batch, :]

                # Add the data to each batch i.e, i will vary from 0 to 29 => 30 batches
                encoder_inputs[i, :, 0:input_size] = data_sel[0:config.SEQ_LENGTH_IN - 1, :]
                decoder_inputs[i, 0, 0:input_size] = data_sel[-1, :]

            for i in range(config.BATCH_SIZE):
                the_key = batch_keys[i]

                # Select the data around the sampled points
                data_sel = real_data[the_key][
                           config.SEQ_LENGTH_IN + sub_batch: config.SEQ_LENGTH_IN + sub_batch + config.SEQ_LENGTH_OUT,
                           :]
                # Add the data to each batch i.e, i will vary from 0 to 29 => 30 batches
                real_decoder_outputs[i, :, :] = data_sel

            pred_outputs = pred_model.step(sess, encoder_inputs, decoder_inputs, decoder_outputs[:, :, :34],
                                           forward_only=True, pred=True)
            final_pred_outputs = data_utils.post_process(pred_outputs, data_mean, data_std)
            final_pred_outputs = np.stack(final_pred_outputs, axis=1)  # modify dimensions to 30x25x35
            # real decoder output dimension is 30x25x35

            # L2 error or Euclidean Distance
            # error = np.sqrt(np.sum(np.square(np.subtract(real_decoder_outputs, final_pred_outputs)),2))  # 30x25
            # pred_test_ms_loss += np.mean(error, 0) # dim 25
            # pred_test_loss += np.mean(error) # average test loss

            # MAE error
            error = np.mean(np.absolute(np.subtract(real_decoder_outputs, final_pred_outputs)), 2)
            pred_test_ms_loss += np.mean(error, 0)
            pred_test_loss += np.mean(error)

        avg_pred_test_loss = pred_test_loss / sub_batches
        print('total prediction test loss : {}'.format(avg_pred_test_loss))

        avg_pred_test_ms_loss = pred_test_ms_loss / sub_batches

        print_results(avg_pred_test_ms_loss)


def test_mogaze():
    config.TEST_MOGAZE = True
    if config.TEST_LOAD <= 0:
        raise (ValueError, "Must give an iteration to read parameters from")

    check_action(config.ACTION)

    # Use the CPU if asked to
    device_count = {"GPU": 0} if config.USE_CPU else {"GPU": 1}

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1, allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count=device_count)) as sess:

        tf.set_random_seed(0)
        random.seed(0)
        np.random.seed(0)

        # === Create the model ===
        print("Creating %d layers of %d units." % (config.NUM_LAYERS, config.SIZE))
        # sampling     = True
        pred_model = create_model(sess, config.TEST_LOAD)
        print("Model created")

        # Load data_mean and std_dev
        data_mean = np.genfromtxt(os.path.join(config.MOGAZE_NORM_STAT_DIR, 'data_mean.csv'), delimiter=',')
        data_std = np.genfromtxt(os.path.join(config.MOGAZE_NORM_STAT_DIR, 'data_std.csv'), delimiter=',')

        # load real data as ground truth
        real_data, _ = utils.data_generation.data_utils.load_mogaze_data(config.MOGAZE_DATA_DIR, actions=['p2_1'],
                                                                         limit=19500)

        # load pred data for prediction
        pred_data = copy.deepcopy(real_data)

        # Normalize -- subtract mean of train data, divide by stdev of train data
        normed_pred_data = data_utils.normalize_data(pred_data, data_mean, data_std)

        # Make prediction
        mogaze_size = config.MOGAZE_SIZE if config.AVOID_GOAL else config.MOGAZE_SIZE + config.GOAL_SIZE  # dim in one time step of qpos data
        real_mogaze_size = config.MOGAZE_SIZE
        encoder_inputs = np.zeros((config.BATCH_SIZE, config.SEQ_LENGTH_IN - 1, mogaze_size), dtype=float)
        decoder_inputs = np.zeros((config.BATCH_SIZE, config.SEQ_LENGTH_OUT, mogaze_size), dtype=float)
        decoder_outputs = np.zeros((config.BATCH_SIZE, config.SEQ_LENGTH_OUT, mogaze_size), dtype=float)
        real_decoder_outputs = np.zeros((config.BATCH_SIZE, config.SEQ_LENGTH_OUT, real_mogaze_size), dtype=float)

        sub_batches = config.MOGAZE_TEST_SUB_BATCH_SIZE
        pred_test_loss = 0
        pred_test_ms_loss = 0
        for sub_batch in range(sub_batches):
            # send 50 data in each loop for prediction
            batch_keys = list(normed_pred_data.keys())

            for i in range(config.BATCH_SIZE):
                # load sub batch to predict
                the_key = batch_keys[i]

                # Select the data around the sampled points
                data_sel = normed_pred_data[the_key][
                           sub_batch * config.SEQ_LENGTH_IN: sub_batch * config.SEQ_LENGTH_IN + config.SEQ_LENGTH_IN,
                           :mogaze_size]
                # Add the data to each batch i.e, i will vary from 0 to 29 => 30 batches
                encoder_inputs[i, :, 0:mogaze_size] = data_sel[0:config.SEQ_LENGTH_IN - 1, :mogaze_size]
                decoder_inputs[i, 0, 0:mogaze_size] = data_sel[-1, :mogaze_size]

            for i in range(config.BATCH_SIZE):
                # load sub batch real data corresponding to prediction (ground truths)
                the_key = batch_keys[i]

                # Select the data around the sampled points
                data_sel = real_data[the_key][
                           sub_batch * config.SEQ_LENGTH_IN + config.SEQ_LENGTH_IN: sub_batch * config.SEQ_LENGTH_IN + config.SEQ_LENGTH_IN + config.SEQ_LENGTH_OUT,
                           :mogaze_size]
                # Add the data to each batch i.e, i will vary from 0 to 29 => 30 batches
                real_decoder_outputs[i, :, :] = data_sel[:, :66]

            pred_outputs = pred_model.step(sess, encoder_inputs, decoder_inputs, decoder_outputs[:, :, :66],
                                           forward_only=True, pred=True)
            final_pred_outputs = data_utils.post_process_mogaze(pred_outputs, data_mean, data_std)
            final_pred_outputs = np.stack(final_pred_outputs, axis=1)  # modify dimension to dimensions to 30x25x66
            # real decoder output dimension is 30x25x35

            # pred_test_loss += np.sqrt(np.mean(np.square(np.subtract(real_decoder_outputs, final_pred_outputs))))
            # pred_test_ms_loss += np.sqrt(np.mean(np.square(np.subtract(real_decoder_outputs, final_pred_outputs)), axis=(0,2)))  # loss at each timestep(1TS = 40ms)

            # MAE
            pred_test_loss += np.mean(np.absolute(np.subtract(real_decoder_outputs, final_pred_outputs)))
            pred_test_ms_loss += np.mean(np.absolute(np.subtract(real_decoder_outputs, final_pred_outputs)),
                                         axis=(0, 2))

        avg_pred_test_loss = pred_test_loss / sub_batches
        print('total prediction test loss : {}'.format(avg_pred_test_loss))

        avg_pred_test_ms_loss = pred_test_ms_loss / sub_batches

        print_results(avg_pred_test_ms_loss)


if __name__ == "__main__":
    if config.TEST_MOGAZE:
        test_mogaze()
    else:
        test()
