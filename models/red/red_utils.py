from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from experiments import config
from models.red import seq2seq_model


def get_dirs(*_):
    train_dir = config.CKPT_DIR
    summaries_dir = os.path.normpath(os.path.join(train_dir, "log"))  # Directory for TB summaries
    return train_dir, summaries_dir


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


def load_model(flags=None, use_cpu=False, ckpt_dir=None, timestep=None, sampling=False):
    """Create model and initialize or load parameters in session."""
    if flags is None:
        try:
            # delete any existing flags
            del_all_flags(tf.flags.FLAGS)
        except:
            pass
        flags = config.RED_FLAGS_FN()
    train_dir, summaries_dir = get_dirs(flags)
    if ckpt_dir is None:
        ckpt_dir = config.CKPT_DIR
    if timestep is None:
        timestep = config.TEST_LOAD
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
