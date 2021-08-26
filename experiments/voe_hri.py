import os
import sys

project_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_dir)
os.environ['PATH'] += os.pathsep + project_dir

import click

from experiments.config import ActionType

# -------------------------
from experiments import config
from utils.data_hri_vis.hri_utils import build_env
from utils.evaluation import hri_gen_qpos_for_hulls, chull_viz_vol_occc_error


@click.command()
@click.option("--architecture", "-a", help='select one architecture: red OR prednet', default="prednet")
@click.option("--action", help='select one action: {}'.format(
    [ActionType.CO_EXISTING, ActionType.CO_OPERATION, ActionType.NOISE, ActionType.COMBINED]),
              default=ActionType.COMBINED)
@click.option("--avoid_goal", help='True for avoiding goal', default=False)
@click.option("--stage_1", help='Execute Prediction Stage', default=True)
@click.option("--stage_2", help='Execute Volumetric Occupancy Error Stage', default=True)
def run(architecture, action=None, avoid_goal=False, stage_1=True, stage_2=True):
    print("running, {} stage 1: {}, stage 2: {}, avoid_goal {}, scenarios {}".format(architecture, stage_1, stage_2,
                                                                                     avoid_goal, action))
    config.ARCHITECTURE = architecture
    config.AVOID_GOAL = avoid_goal
    config.VIZ_VOE_CHULL = False
    config.update_experiments_dir()

    # utils.evaluation.archive.chull_viz_vol_occc_error.play()
    if action is None or action == ActionType.COMBINED:
        scenarios = ["co-existing", "co-operating", "noise"]
    else:
        if isinstance(action, str):
            scenarios = [action]
        else:
            scenarios = action

    if stage_1:
        # uses trained model to generate predictions
        hri_gen_qpos_for_hulls.run(scenarios)
        config.reset_tf()
    else:
        pass

    if stage_2:
        chull_viz_vol_occc_error.play(build_env, force_update=False, scenarios=scenarios)


if __name__ == '__main__':
    run()

# run("prednet", avoid_goal=False, stage_1=True, scenarios = [ "co-operating"]) # terminal 1 # prednet with goal, prednet without goal are available on the other pc
# run("prednet", avoid_goal=False, stage_1=False, scenarios = [ "noise"]) # terminal 2 # prednet with goal, prednet without goal are available on the other pc
# run("prednet", avoid_goal=True, stage_1=False, scenarios = [ "co-operating"])  # terminal 3 #prednet with goal, prednet without goal are available on the other pc
# run("prednet", avoid_goal=True, stage_1=False, scenarios=["noise"])  # terminal 4 #prednet with goal, prednet without goal are available on the other pc

# run("red", avoid_goal=False, stage_1=True, stage_2=True, scenarios=["co-existing"])
# run("red", avoid_goal=False, stage_1=False, stage_2=True, scenarios=["co-operating"])
# run("red", avoid_goal=False, stage_1=True, stage_2=True, scenarios=["noise"])

# =====================================================RED MSM==============================================================
# ============================"noise"==============================
# voc_avg_step_error =  53.89098460858095
# prediction_step        |    80         |   160         |   320         |   400         |   600         |   720         |   880         |  1000         |
# voc_error ± 0.5std (%) | 89.126 ± 20.85| 86.775 ± 21.24| 76.743 ± 15.08| 73.325 ± 12.19| 66.736 ± 15.58| 62.262 ± 15.47| 56.343 ± 10.17| 53.891 ± 11.51|
# voc_error ± 0.5std (%) &$ 89.1 \pm 20.9$&$ 86.8\pm 21.2$&$ 76.7 \pm 15.1$&$ 73.3 \pm 12.2$&$ 66.7 \pm 15.6$&$ 62.2 \pm 15.5$&$ 56.3 \pm 10.2$&$ 53.9 \pm 11.5$\\
# ============================"co-operating"==============================
# voc_avg_step_error =  25.76372109309117
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
# voc_error ± std (%) | 59.014 ± 18.69| 50.004 ± 22.12| 25.923 ± 28.32| 22.356 ± 16.18| 23.396 ± 19.40| 22.583 ± 13.79| 24.564 ± 14.74| 25.764 ± 12.80|
# voc_error ± std (%) &$ 59.0 \pm 18.7$&$ 50.0 \pm 22.1$&$ 25.9 \pm 28.3$&$ 22.4 \pm 16.2$&$ 23.4 \pm 19.4$&$ 22.5 \pm 13.8$&$ 24.5 \pm 14.7$&$ 25.8 \pm 12.8$\\
# ============================"co-existing"==============================
# voc_avg_step_error =  21.37874980746577
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
# voc_error ± std (%) | 56.184 ± 21.77| 48.856 ± 23.27| 24.726 ± 11.28| 23.423 ± 14.22| 20.435 ± 8.05| 20.925 ± 13.28| 20.204 ± 9.64| 21.379 ± 8.63|
# voc_error ± std (%) | $56.2 \pm 21.8$ & $48.9 \pm 23.3$ & $24.7 \pm 11.3$ & $23.4 \pm 14.2$ & $20.4 \pm 8.1$ & $20.9 \pm 13.3$ & $20.2 \pm 9.6$ & $21.4 \pm 8.6$ \\


# =====================================================PredNet MSM==========================================================
# ============================"noise"==============================
# voc_avg_step_error =  18.386723713546743
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
# voc_error ± std (%) | 8.467 ± 20.06| 12.791 ± 28.77| 10.955 ± 19.60| 12.288 ± 23.20| 14.916 ± 22.81| 14.574 ± 17.30| 16.645 ± 13.77| 18.387 ± 16.29|
# voc_error ± std (%) &$ 8.5 \pm 20.1$&$ 12.8 \pm 28.8$&$ 11.0 \pm 19.6$&$ 12.3 \pm 23.2$&$ 14.9 \pm 22.8$&$ 14.6 \pm 17.3$&$ 16.6 \pm 13.8$&$ 18.4 \pm 16.3$\\
# ============================"co-operating"==============================
# voc_avg_step_error =  15.07198288388537
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
# voc_error ± std (%) | 12.896 ± 36.43| 12.526 ± 25.65| 12.975 ± 22.34| 12.255 ± 19.39| 11.765 ± 9.83| 13.385 ± 9.78| 13.458 ± 5.99| 15.072 ± 9.66|
# voc_error ± std (%) $&$ 12.9 \pm 36.4$&$ 12.5 \pm 25.7$&$ 13.0 \pm 22.3$&$ 12.3 \pm 19.4$&$ 11.8 \pm 9.8$&$ 13.4 \pm 9.8$&$ 13.5 \pm 6.0$&$ 15.1 \pm 9.7$\\
# ============================"co-existing"==============================
# voc_avg_step_error =  15.143050917051056
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
# voc_error ± std (%) | 12.707 ± 37.20| 9.470 ± 19.04| 13.274 ± 20.49| 14.668 ± 23.43| 12.512 ± 15.09| 12.624 ± 10.46| 15.029 ± 13.09| 15.143 ± 12.31|
# voc_error ± std (%) $&$ 12.7 \pm 37.2$&$ 9.5 \pm 19.0$&$ 13.3 \pm 20.5$&$ 14.7 \pm 23.4$&$ 12.5 \pm 15.1$&$ 12.6 \pm 10.5$&$ 15.0 \pm 13.1$&$ 15.1 \pm 12.3$\\


# =====================================================PredNet MSMwG==========================================================
# ============================"noise"==============================
# voc_avg_step_error = 17.59039606551205
# prediction_step        |    80         |   160         |   320         |   400         |   600         |   720         |   880         |  1000         |
# voc_error ± 0.5std (%) | 8.622 ± 20.89| 15.121 ± 39.35| 10.321 ± 14.96| 14.358 ± 21.53| 12.899 ± 10.77| 14.466 ± 13.06| 15.761 ± 10.62| 17.590 ± 11.84|
# voc_error ± 0.5std (%) &$ 8.6 \pm 20.9$&$ 15.1 \pm 39.4$&$ 10.3 \pm 15.0$&$ 14.4 \pm 21.5$&$ 12.9 \pm 10.8$&$ 14.5 \pm 13.1$&$ 15.8 \pm 10.6$&$ 17.6 \pm 11.8$\\
# ============================"co-operating"==============================
# voc_avg_step_error =  11.473376380300131
# prediction_step        |    80         |   160         |   320         |   400         |   600         |   720         |   880         |  1000         |
# voc_error ± 0.5std (%) | 12.204 ± 31.57| 11.088 ± 24.15| 12.836 ± 22.57| 14.558 ± 24.50| 10.084 ± 8.94| 11.358 ± 10.83| 10.967 ± 7.58| 11.473 ± 8.53|
# voc_error ± 0.5std (%) $&$ 12.2 \pm 31.6$&$ 11.088 \pm 24.2$&$ 12.8 \pm 22.6$&$ 14.6 \pm 24.5$&$ 10.1 \pm 8.9$&$ 11.4 \pm 10.8$&$ 11.0 \pm 7.6$&$ 11.5 \pm 8.5$\\
# ============================"co-existing"==============================
# voc_avg_step_error =  13.357506084928701
# prediction_step        |    80         |   160         |   320         |   400         |   600         |   720         |   880         |  1000         |
# voc_error ± 0.5std (%) | 11.008 ± 27.95| 11.188 ± 23.48| 10.855 ± 17.51| 11.005 ± 16.62| 10.910 ± 15.45| 12.233 ± 14.93| 11.162 ± 7.35| 13.358 ± 11.44|
# voc_error ± 0.5std (%) $&$ 11.0 \pm 28.0$&$ 11.2 \pm 23.5$&$ 10.9 \pm 17.5$&$ 11.0 \pm 16.6$&$ 10.9 \pm 15.5$&$ 12.2 \pm 14.9$&$ 11.2 \pm 7.4$&$ 13.4 \pm 11.4$\\
# run("red")

# ============================================NEW prednet MsM===========================================================

# ==============co-existing============================
# voc_avg_step_error =  12.265122559982968
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
# voc_error ± std (%)  9.03 | 9.70 | 11.32 | 11.87 | 12.98 | 14.11 | 15.28 | 16.24 |
# ==============co-operating============================
# voc_avg_step_error =  12.265122559982968
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
# voc_error ± std (%)  8.69 & 9.12 & 10.52 & 10.64 & 11.15 & 11.61 & 12.18 & 12.27 \\
# ==============Noise===================================
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
#                     9.26 & 9.93 & 10.31 & 10.84 & 11.15 & 12.22 & 13.27 & 14.08 \\

# ============================================NEW prednet MsMwG=========================================================

# ==============co-existing============================
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
# voc_error (%)       9.43 | 9.09 | 11.77 | 11.52 | 13.47 | 13.36 | 14.95 | 16.60 |

# ==============co-operating============================
# voc_avg_step_error =  11.172541705667783
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
# voc_error ± std (%)  9.14 & 9.81 & 9.98 & 10.07 & 10.40 & 10.66 & 10.85 & 11.18 \\

# ==============Noise============================

# # prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
#                        9.68 & 10.81 & 11.61 & 11.62 & 11.56 & 12.65 & 13.50 & 14.70 \\


# ==========================================RED========================================================
# ==============Co-existing============================
# voc_avg_step_error =  41.73231900783868
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
# voc_error ± std (%) | 71.895 & 66.920 & 48.586 & 47.301 & 45.140 & 43.568 & 42.496 & 41.732 \\
# ==============Co-operating============================
# voc_avg_step_error =  32.17721419484325
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
# voc_error ± std (%) | 72.21 & 67.45 & 39.73 & 38.42 & 35.02 & 33.28 & 32.85 & 32.18 \\
# ==============Noise============================
# voc_avg_step_error =  41.73231900783868
# prediction_step  |    80 |   160 |   320 |   400 |   600 |   720 |   880 |  1000 |
# voc_error ± std (%) | 71.90 & 66.92 & 48.60 & 47.30 & 45.14 & 43.57 & 42.50 & 41.73
