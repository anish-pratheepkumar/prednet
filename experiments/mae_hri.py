import os
import sys

project_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_dir)
os.environ['PATH'] += os.pathsep + project_dir

import click

from experiments.config import ActionType

# -------------------------
import tensorflow as tf

from experiments import config

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


@click.command()
@click.option("--architecture", "-a", help='select one architecture: red OR prednet', default="prednet")
@click.option("--action", help='select one action: {}'.format(
    [ActionType.CO_EXISTING, ActionType.CO_OPERATION, ActionType.NOISE, ActionType.COMBINED]),
              default=ActionType.COMBINED)
@click.option("--avoid_goal", help='True for avoiding goal', default=False)
def run(architecture, action, avoid_goal=False):
    config.ARCHITECTURE = architecture
    config.AVOID_GOAL = avoid_goal
    config.ACTION = action
    config.TEST_DATA = "pred_test_noise"
    config.update_experiments_dir()
    if action == "combined":
        scenarios = ["co-existing", "co-operating", "noise"]
    else:
        scenarios = [action]
    for scenario in scenarios:
        print("Scenario: ", scenario)
        if architecture == "prednet":
            from models.prednet.test_prednet import test
            test(scenario)
        else:
            from models.red.HRI_Scenario.HRI_translate import create_flags, test as test_red
            flags = create_flags()
            test_red(flags, config.TEST_LOAD, scenario)
            try:
                tf.flags.FLAGS.remove_flag_values(tf.flags.FLAGS.flag_values_dict())
                tf.flags.FLAGS.__delattr__()
            except:
                pass
        config.reset_tf()


#
# run("prednet", action="combined", avoid_goal=True)
# run("prednet", action="combined", avoid_goal=False)
# run("prednet", action="co-existing", avoid_goal=False)
# run("prednet", action="co-operating", avoid_goal=False)
# run("prednet", action="noise", avoid_goal=False)
#
# run("red", "combined", False)

if __name__ == '__main__':
    run()
