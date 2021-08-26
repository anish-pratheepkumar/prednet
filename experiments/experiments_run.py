import json
import os.path
import sys
import time
from pathlib import Path
from types import ModuleType

import click

sys.path.insert(0, os.path.abspath(Path(__file__).parent.parent))

from experiments import config

experiment_names = {
    "msm_with_goal": None,  # PREDNET, red  TODO currently running;
    "msm_without_goal": None,  # TODO
    "ssm_co-existing": None,
    "ssm_co-operating": None,
    "ssm_noise": None,
    "mogaze": None,  # PREDNET, red
    "mogaze_without_goal": None  # PREDNET
}


def prepare_params(experiment_name):
    config.ACTION = 'combined'  # default scenario type
    if experiment_name == "msm_with_goal":
        config.AVOID_GOAL = False
    elif experiment_name == "msm_without_goal":
        config.AVOID_GOAL = True
    elif "ssm" in experiment_name:
        scenario_type = experiment_name.split("_")[-1]
        config.AVOID_GOAL = False
        config.ACTION = scenario_type
    elif experiment_name == "mogaze":
        config.ACTION = 'p1_1'
        config.TRAIN_MOGAZE = True


@click.command()
@click.option("--architecture", "-a", help='select architecture: red, PREDNET', default="red")
@click.option("--experiment", "-e", help='select experiment: {}'.format(experiment_names.keys()),
              default="msm_with_goal")
def start_experiment(architecture, experiment):
    architecture, experiment = architecture.lower(), experiment.lower()
    prepare_params(experiment)
    config.ARCHITECTURE = architecture
    config.update_experiments_dir()
    start_time = time.time()

    if "mogaze" in experiment.lower():
        config.TRAIN_MOGAZE = True
        config.ACTION = "p1_1"
        config.ITERATIONS = 6000
        config.AVOID_GOAL = False if experiment.lower() == "mogaze" else True
        config.update_experiments_dir()
        config_save_path = os.path.join(config.TRAIN_DIR, "config.txt")
    else:
        config_dir = config.TRAIN_DIR
        config.TRAIN_MOGAZE = False
        config_save_path = os.path.join(config_dir, "config.txt")

    if architecture.lower() == "prednet":
        if "mogaze" in experiment.lower():
            from models.prednet import train_prednet
            dict_tf = config.__dict__.copy()
            for k in list(dict_tf.keys()):
                if k[0] == "_" or k in ["Path", "os", "update_experiments_dir"]:
                    dict_tf.pop(k)
            save_config(dict_tf, config_save_path)
            train_prednet.run()
            dict_tf["total_time"] = time.time() - start_time
            save_config(dict_tf, config_save_path)
        else:
            from models.prednet import train_prednet
            dict_tf = config.__dict__.copy()
            for k in list(dict_tf.keys()):
                if k[0] == "_" or k in ["Path", "os", "update_experiments_dir"]:
                    dict_tf.pop(k)
            save_config(dict_tf, config_save_path)
            train_prednet.run()
            dict_tf["total_time"] = time.time() - start_time
            save_config(dict_tf, config_save_path)

    elif architecture.lower() == "red":
        if experiment.lower() == "mogaze":
            from models.red.Mogaze import mogaze_translate
            flags = mogaze_translate.create_flags()
            import tensorflow as tf
            dict_tf = {k: v.value for k, v in flags.__flags.items()}
            config_save_path = os.path.join(mogaze_translate.get_dirs(flags)[0], "config.txt")
            save_config(dict_tf, config_save_path)
            mogaze_translate.train(flags)
            dict_tf["total_time"] = time.time() - start_time
            save_config(dict_tf, config_save_path)
        else:
            from models.red.HRI_Scenario import HRI_translate
            flags = HRI_translate.create_flags()
            dict_tf = {k: v.value for k, v in flags.__flags.items()}
            config_save_path = os.path.join(HRI_translate.get_dirs()[0], "config.txt")
            save_config(dict_tf, config_save_path)
            HRI_translate.tf.app.run(main=HRI_translate.main)
            dict_tf["total_time"] = time.time() - start_time
            save_config(dict_tf, config_save_path)
    else:
        raise Exception("Check inputs: ", architecture, experiment)
    print("Total time: {}h".format(dict_tf["total_time"] / 3600))
    print("finished: ", architecture, experiment)


def save_config(config_dict, config_save_path):
    if not os.path.exists(os.path.dirname(config_save_path)):
        os.makedirs(os.path.dirname(config_save_path))
    with open(config_save_path, mode="w", encoding="utf8") as fp:
        # remove all functions
        config_dict_to_save = config_dict.copy()
        for k, v in config_dict.items():
            if callable(v) or isinstance(v, ModuleType):
                config_dict_to_save.pop(k)
        json.dump(config_dict_to_save, fp, indent=4, sort_keys=True)


if __name__ == '__main__':
    start_experiment()
