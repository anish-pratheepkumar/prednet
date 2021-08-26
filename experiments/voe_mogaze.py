import os
import sys

import click

project_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_dir)
os.environ['PATH'] += os.pathsep + project_dir

# -------------------------
from experiments import config


@click.command()
@click.option("--architecture", "-a", help='select one architecture: red OR prednet', default="prednet")
@click.option("--stage_1", help='Execute Prediction Stage', default=True)
@click.option("--avoid_goal", help='True for avoiding goal', default=False)
def run(architecture, stage_1=True, avoid_goal=False):
    print("running, {} stage 1: {}, avoid_goal {}".format(architecture, stage_1, avoid_goal))
    config.ARCHITECTURE = architecture
    config.ACTION = "p1_1"
    config.VIZ_VOE_CHULL = False
    config.AVOID_GOAL = avoid_goal
    from utils.evaluation.mogaze_vis import apply_mogaze_settings
    apply_mogaze_settings(avoid_goal=avoid_goal)

    from utils.evaluation import mogaze_gen_qpos_for_hulls

    if stage_1:
        mogaze_gen_qpos_for_hulls.run()
        config.reset_tf()

    from utils.data_mogaze_vis.mogaze_utils import build_env
    chull_viz_vol_occc_error.play(build_env, force_update=False, scenarios=[""])


from utils.evaluation import chull_viz_vol_occc_error

if __name__ == '__main__':
    run()

# =====================================================RED==============================================================
# voc_avg_step_error =  52.050476178331344
# prediction_step        |    80         |   160         |   320         |   400         |   600         |   720         |   880         |  1000         |
# voc_error ± 0.5std (%) | 33.354 ± 8.13| 36.354 ± 8.60| 40.999 ± 9.90| 42.783 ± 10.25| 46.690 ± 11.05| 48.500 ± 11.27| 50.737 ± 11.45| 52.050 ± 11.39|
# voc_error ± 0.5std (%) &$ 33.4 \pm 8.1$&$ 36.4 \pm 8.6$&$ 41.0 \pm 9.9$&$ 42.8 \pm 10.3$&$ 46.7 \pm 11.1$&$ 48.5 \pm 11.3$&$ 50.7 \pm 11.5$&$ 52.1 \pm 11.4$\\
# End time: 1624159094.5830145
# =====================================================PredNet==========================================================
# prediction_step     |    80        |   160        |   320        |   400        |   600         |   720         |   880         |  1000         |
# voc_error ± std (%) | 14.832 ± 6.68| 17.682 ± 6.19| 23.536 ± 8.29| 26.041 ± 9.50| 31.637 ± 11.02| 34.106 ± 11.63| 37.335 ± 12.23| 39.510 ± 12.47|
# voc_error ± std (%) $&$ 14.8 \pm 6.7$&$ 17.7 \pm 6.19$&$ 23.6 \pm 8.3$&$ 26.1 \pm 9.5$&$ 31.6 \pm 11.0$&$ 34.1 \pm 11.6$&$ 37.3 \pm 12.2$&$ 39.5 \pm 12.5$\\
# End time: 1624193836.6961882
# run("red")
# run("red", True, False)
# run("prednet", True, False)  # Prednet MSM
# run("prednet", False, True) #Prednet MSMwg

# ==================================NEW======================================================================
# ==================Prednet MSM=============
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
# voc_error ± std (%) | 15.71 & 18.57 & 23.97 & 26.74 & 32.06 & 34.44 & 37.70 & 39.82 \\
# ==================Prednet MSMwG=============
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
# voc_error & 15.10 & 17.54 & 23.73 & 26.47 & 33.17 & 35.48 & 38.32 & 40.31 \\
# ==================RED=======================
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
# voc_error & 32.94 & 35.78 & 40.53 & 42.36 & 46.36 & 48.02 & 50.21 & 51.56
