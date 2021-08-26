import os
import sys

import click

project_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_dir)
os.environ['PATH'] += os.pathsep + project_dir

# -------------------------
import tensorflow as tf

from experiments import config

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


@click.command()
@click.option("--architecture", "-a", help='select one architecture: red OR prednet', default="prednet")
@click.option("--avoid_goal", "-ag", help='True for avoiding goal', default=False)
def run(architecture, avoid_goal=False):
    print("running MAE Mogaze, {}, avoid_goal {}".format(architecture, avoid_goal))
    config.ARCHITECTURE = architecture
    config.ACTION = "p1_1"
    config.VIZ_VOE_CHULL = False
    config.AVOID_GOAL = avoid_goal
    from utils.evaluation.mogaze_vis import apply_mogaze_settings
    apply_mogaze_settings(avoid_goal=avoid_goal, test=True)
    if architecture == "prednet":
        from models.prednet.test_prednet import test_mogaze
        test_mogaze()
    else:
        # TODO test
        from models.red.Mogaze.mogaze_translate import create_flags, test as test_red
        flags = create_flags()
        test_red(flags, config.TEST_LOAD)
        try:
            tf.flags.FLAGS.remove_flag_values(tf.flags.FLAGS.flag_values_dict())
            tf.flags.FLAGS.__delattr__()
        except:
            pass
    config.reset_tf()


# run("prednet", False)  # avoid_goal =True is Prednet MSMwg
# run("prednet", True)  # avoid_goal =True is Prednet MSMwg
# run("red", False)

# =======================================================MAE Mogaze=================================================
# ==================================================PredNet MSMwG====================================================
# Loading model /home/mae/git/mae_ap/motion_prediction/experiments/results/models/prednet/p1_1_avoid_goal/checkpoint-6000
# milliseconds     | 83.33 | 166.66 | 333.33 | 416.66 | 625.00 | 750.00 | 916.66 | 1041.66 |
# walking          & 0.055 & 0.090 & 0.139 & 0.154 & 0.185 & 0.194 & 0.213 & 0.225 \\

# ==================================================PredNet MSM====================================================
# milliseconds     | 83.33 | 166.66 | 333.33 | 416.66 | 625.00 | 750.00 | 916.66 | 1041.66 |
# walking          | 0.048 | 0.080 | 0.126 | 0.141 | 0.167 | 0.175 | 0.190 | 0.203 |

if __name__ == '__main__':
    run()
