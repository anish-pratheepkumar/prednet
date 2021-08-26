import os
import sys

project_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_dir)
os.environ['PATH'] += os.pathsep + project_dir

from pathlib import Path


class ActionType:
    CO_EXISTING = "co-existing"
    CO_OPERATION = "co-operating"
    COMBINED = "combined"
    NOISE = "noise"


def reset_tf():
    import tensorflow as tf
    tf.reset_default_graph()
    # tf.flags.FLAGS.remove_flag_values(tf.flags.FLAGS.flag_values_dict())
    # tf.flags.FLAGS.__delattr__()
    #
    # def del_all_flags(FLAGS):
    #     flags_dict = FLAGS._flags()
    #     keys_list = [keys for keys in flags_dict]
    #     for keys in keys_list:
    #         FLAGS.__delattr__(keys)
    #
    # del_all_flags(tf.flags.FLAGS)


BLENDER_OR_OPENSCAD_PATH = r"D:\Program Files\Blender 2.92"  # None

if BLENDER_OR_OPENSCAD_PATH is not None:
    import sys, os

    sys.path.insert(0, BLENDER_OR_OPENSCAD_PATH)
    os.environ['PATH'] += os.pathsep + BLENDER_OR_OPENSCAD_PATH

config_root_dir = os.path.dirname(os.path.abspath(__file__))  # added
# for any parameter related mogaze data 'MOGAZE' is mentioned in the variable name

# set  whether to consider goal input for prednet_run
AVOID_GOAL = False

# Hyperparameter
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY_FACTOR = 0.95
LEARNING_RATE_STEP = 10000
MAX_GRADIENT_NORM = 8
BATCH_SIZE = 30
ITERATIONS = 3600

# Architecture - rnn_GRU
ARCHITECTURE = "prednet"  # share encoder decoder parameters
SIZE = 1024
NUM_LAYERS = 1
SEQ_LENGTH_IN = 50
SEQ_LENGTH_OUT = 25
RESIDUAL_VELOCITIES = True
LOSS_TO_USE = 'sampling_based'

# Directories
ROOT_DIR = os.path.normpath(Path(__file__).parent.parent)
DATA_DIR = os.path.normpath(os.path.join(ROOT_DIR, 'data/hri_data/'))
MOGAZE_DATA_DIR = os.path.normpath(os.path.join(ROOT_DIR, 'data/mogaze_data/'))
EXP_DIR = os.path.normpath(os.path.join(ROOT_DIR, 'experiments/results/'))
ACTION = 'combined'  # or 'noise', 'co-existing', 'co-operating'

# train parameters
TRAIN_MOGAZE = False
VAL_EVERY = 100
SAVE_EVERY = 100
USE_CPU = False
TRAIN_LOAD = 0
SUB_BATCH_SIZE = 13
MOGAZE_SUB_BATCH_SIZE = 68
MOGAZE_VAL_SUB_BATCH_SIZE = 22

# test parameters
TEST_MOGAZE = False
TEST_LOAD = 1500  # for RED 1500, for PredNet 1800,
TEST_DATA = 'pred_test_noise'
TEST_SUB_BATCH_SIZE = 325
MOGAZE_TEST_SUB_BATCH_SIZE = 2

# data
OUTPUT_QPOS_SIZE = 35
QPOS_SIZE = 34 if AVOID_GOAL else 37
HUMAN_SIZE = 34  # after converting root_orientation quat -> euler
MOGAZE_SIZE = 66
GOAL_SIZE = 3

# ------------------------------------
# variables for chull viz. and vol. occ. error computation
# ------------------------------------

EPISODES = 30
PREDICTION_STEPS = 325  # removed,
TOTAL_STEPS = 350  # 325+25
# CHULL_SCENARIO = 'co-existing'
VIZ_QPOS_CHULL = False
VIZ_VOE_CHULL = True

# ------------------------------------
TRAIN_DIR = None
SUMMARIES_DIR = None
CKPT_DIR = None
CHULL_BASE_DIR = None
DIR_PRED_CHULL_QPOS = None
DIR_DRL_CHULL_QPOS = None
DIR_PRED = None
DIR_DRL = None
DIR_INTERSECTION = None
MOGAZE_NORM_STAT_DIR = None
NORM_STAT_DIR = None
HRI_TEST_DATA_DIR = None

CREATE_MODEL_HRI_FN = None
CREATE_MODEL_MOGAZE_FN = None

LOAD_MODEL_FN = None  # function for loading PredNet or RED model
RED_FLAGS_FN = None

HRI_DATA_PATH = os.path.join(DATA_DIR, "hri_scenarios.h5")


def update_experiments_dir():
    global TRAIN_DIR, SUMMARIES_DIR, CKPT_DIR, CHULL_BASE_DIR, DIR_PRED_CHULL_QPOS, DIR_DRL_CHULL_QPOS, DIR_PRED, \
        DIR_DRL, DIR_INTERSECTION, QPOS_SIZE, MOGAZE_NORM_STAT_DIR, NORM_STAT_DIR, HRI_TEST_DATA_DIR, \
        CREATE_MODEL_HRI_FN, CREATE_MODEL_MOGAZE_FN, LOAD_MODEL_FN, RED_FLAGS_FN, TEST_LOAD
    avoid_goal_str = "_avoid_goal" if AVOID_GOAL else ""
    TRAIN_DIR = os.path.normpath(os.path.join(EXP_DIR,
                                              "models",
                                              ARCHITECTURE,
                                              '{}{}'.format(ACTION, avoid_goal_str),
                                              # 'out_{0}'.format(SEQ_LENGTH_OUT),
                                              # 'iterations_{0}'.format(ITERATIONS),
                                              # "tied",
                                              # LOSS_TO_USE,
                                              # 'depth_{0}'.format(NUM_LAYERS),
                                              # 'size_{0}'.format(SIZE),
                                              # 'lr_{0}'.format(LEARNING_RATE),
                                              # 'residual_vel' if RESIDUAL_VELOCITIES else 'not_residual_vel'
                                              ))
    SUMMARIES_DIR = os.path.normpath(os.path.join(TRAIN_DIR, "log"))  # Directory for TB summaries
    CKPT_DIR = TRAIN_DIR  # os.path.normpath(TRAIN_DIR, 'model_ckpt')

    if (ACTION == "combined" or ACTION == "p1_1") and AVOID_GOAL:
        CHULL_BASE_DIR = os.path.join(EXP_DIR, "voe", ARCHITECTURE, ACTION + '_avoid_goal')
    elif AVOID_GOAL:
        CHULL_BASE_DIR = os.path.join(EXP_DIR, "voe", ARCHITECTURE, ACTION, 'avoid_goal')
    else:
        CHULL_BASE_DIR = os.path.join(EXP_DIR, "voe", ARCHITECTURE, ACTION)
    # qpos data dir
    DIR_PRED_CHULL_QPOS = os.path.join(CHULL_BASE_DIR, 'pred_chull_viz', 'qpos')
    DIR_DRL_CHULL_QPOS = os.path.join(CHULL_BASE_DIR, 'drl_chull_viz', 'qpos')
    # mesh_files dir
    DIR_PRED = os.path.join(CHULL_BASE_DIR, 'pred_chull_viz', 'mesh_files')
    DIR_DRL = os.path.join(CHULL_BASE_DIR, 'drl_chull_viz', 'mesh_files')
    DIR_INTERSECTION = os.path.join(CHULL_BASE_DIR, 'intersection_meshes')
    HRI_TEST_DATA_DIR = os.path.join(DATA_DIR, ACTION, 'test_all')
    if ARCHITECTURE == "prednet":
        MOGAZE_NORM_STAT_DIR = os.path.join(MOGAZE_DATA_DIR, 'norm_stat/')
        NORM_STAT_DIR = os.path.join(DATA_DIR, ACTION, 'norm_stat/')
        QPOS_SIZE = 34 if AVOID_GOAL else 37
        from models.prednet.train_prednet import create_model
        CREATE_MODEL_HRI_FN = create_model
        CREATE_MODEL_MOGAZE_FN = create_model
        from models.prednet.prednet_model import PredNet
        LOAD_MODEL_FN = PredNet.load_model
        if ACTION == "p1_1":
            TEST_LOAD = 6000  # for RED 1500, for PredNet 1800,
        elif ACTION == "combined":
            if AVOID_GOAL:
                TEST_LOAD = 3000  # for RED 1500, for PredNet 1800,
            else:
                TEST_LOAD = 1800  # for RED 1500, for PredNet 1800,
        elif ACTION == "co-operating":
            TEST_LOAD = 3500  # for RED 1500, for PredNet 1800,
        elif ACTION == "co-existing":
            TEST_LOAD = 900  # for RED 1500, for PredNet 1800,
        elif ACTION == "noise":
            TEST_LOAD = 1300  # for RED 1500, for PredNet 1800,
        else:
            raise Exception("Unknown action {}".format(ACTION))
    else:
        MOGAZE_NORM_STAT_DIR = os.path.join(MOGAZE_DATA_DIR, 'norm_stat_red/')
        NORM_STAT_DIR = os.path.join(DATA_DIR, ACTION, 'norm_stat_red/')
        QPOS_SIZE = 37
        if ACTION == "p1_1":
            from models.red.Mogaze.mogaze_translate import create_model as red_mogaze_create_model, create_flags
            CREATE_MODEL_MOGAZE_FN = red_mogaze_create_model
            RED_FLAGS_FN = create_flags
            TEST_LOAD = 6000  # for RED 1500, for PredNet 1800,

        else:
            from models.red.HRI_Scenario.HRI_translate import create_model as red_hri_create_model, create_flags
            CREATE_MODEL_HRI_FN = red_hri_create_model
            RED_FLAGS_FN = create_flags
            TEST_LOAD = 1500  # for RED 1500, for PredNet 1800,

        from models.red.red_utils import load_model
        LOAD_MODEL_FN = load_model


update_experiments_dir()
