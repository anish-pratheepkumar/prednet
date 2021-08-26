# PredNet

This repo accompanies the paper _"PredNet: a simple Human Motion Prediction Network for Human-Robot Interaction"_, to be
published in the ETFA 2021.

# Table of Contents

1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Repeating experiments from the paper](#experiments)
4. [Dependencies](#dependencies)
5. [Cite](#cite)
6. [License](#license)
7. [Acknowledgement](#ack)

# 1. Installation <a name="installation"></a>

### A. Core Requirements

1. Install [Mujoco 2.0+](https://www.roboti.us/index.html).
2. Using conda, install this repo's dependencies:

  ```bash
  conda env create -f env_OS.yml
  ``` 

`OS` is either `linux` or `windows`.

3. Activate the conda env before running any of the code below:

```bash
conda activate motion_prediction
```

### B. For calculating Volumetric Occupancy Error (VOE) in Evaluations

1. Install [Blender](https://www.blender.org/download/) OR [OpenSCAD](https://openscad.org/downloads.html)
2. Add the directory of Blender OR OpenSCAD to `$PATH` OR to `BLENDER_OR_OPENSCAD_PATH` in `experiments/config.py`. It
   will accordingly be used by `Trimesh` to calculate the intersection volume necessary for the VOE.

__Note__: Blender 2.92 was tested and used to produce the results in the paper.

# 2. Datasets  <a name="datasets"></a>

Two datasets are supported by this repo:

1. HRI synthetic dataset
2. [Mogaze](https://github.com/humans-to-robots-motion/mogaze) real-world dataset: ### A. HRI synthetic data (
   self-developed in simulation)
   The HRI synthetic data developed in this work are located under: `data/hri_data`. The data contains three scenario
   types:
   co-operating, co-existing and noise. To visualize a scenario, in a terminal with the `motion_prediction` conda env
   active, run:

```bash
python utils/data_hri_vis/hri_utils.py --scenario combined --dataset_type train
```

Parameters are:

- `--scenario` or `-s`: scenario to visualize:
    - `co_operating`: human waiting for the robot to stop, then working directly beside the robot's workspace
    - `co_existing`: human working far away from the robot
    - `noise`: human walking beside the robot
    - `combined`: all of the above
- `--dataset_type` or `-t`: dataset type to use:
    - `train`: data used in training the model
    - `test`: data used in testing the model (scenarios are slightly different than above, please check the paper).

### B. Mogaze data

[Mogaze](https://github.com/humans-to-robots-motion/mogaze) data daily human manipulation scenarios, captured in the
real-word. We currently provide the data from Mogaze in the repo to ease repeating the exepriments results. However, the
data can also be downloaded from the official repo [Mogaze](https://github.com/humans-to-robots-motion/mogaze). Place
any downloaded scenarios in its folder under `data`, e.g., for `p1_1`:

- `data/p1_1/p1_1_gaze_data.hdf5`
- `data/p1_1/p1_1_human_data.hdf5`

To visualize the data, run:

```bash
python utils\data_mogaze_vis\mogaze_utils.py -s p2_1
```

Parameters are:

- `--scenario` or `-s`: scenario to visualize:
    - `p1_1`: mogaze user 1 scenario 1 (used for training)
    - `p2_1`: mogaze user 2 scenario 1 (used for testing)

# 3. Repeating experiments from the paper  <a name="experiments"></a>

### A. Training

Run `experiments/experiments_run.py` with the parameters:

- `--architecture` or `-a`: model architecture to use, either `red` or `prednet`
- `--experiment` or `-e`: experiment to run based. Possible experiment keys are:
    - `msm_with_goal`: HRI Multiple Scenario data (i.e. training includes data from noise, co-existing and co-operating
      scenarios) with considering goal position at the input
    - `msm_without_goal`: same as above but without considering goal position at the input
    - `ssm_co-existing`: HRI Single Scenario data using data only for co-existing with considering goal
    - `ssm_co-operating`: HRI Single Scenario data using data only for co-operating with considering goal
    - `ssm_noise`: HRI Single Scenario data using data only for noise with considering goal
    - `mogaze`: Mogaze data with considering goal at the input
    - `mogaze_without_goal`: same as above but without considering goal at the input

Note: make sure you have your conda env is activated!

#### Examples:

```bash
# Training "PredNet MSM" on HRI data:
python experiments\experiments_run.py -a prednet -e msm_with_goal

# Training "PredNet MSM" on Mogaze data:
python experiments\experiments_run.py -a prednet -e mogaze

# Training "RED" on HRI data:
python experiments\experiments_run.py -a red

# Training "RED" on Mogaze data:
python experiments\experiments_run.py -a red -e mogaze
```

### B. Evaluation (testing)

You can test the trained models with two metrics:

- Mean Absolute Error (MAE)
- Volumetric Occupancy Error (VOE).

__Note__: By default, the configurations are set to repeat the paper results, in `config.py`.

#### For calculating Mean Absolute Error (MAE)

- For HRI: e.g., for calculating MAE for PredNet MSM on all scenario types:

```bash
python experiments/mae_hri.py -a prednet --action combined --avoid_goal False
```

Parameters:

- `--architecture` or `-a`: trained model architecture to use, either `red` or `prednet`
- `--action` or `-ac`: action / scenario type. Possible keys are: co_exsiting, co_operating, noise, combined
- `--avoid_goal` or `-ag`: true for avoiding passing the goal position at the input By default, the trained models
  provided with the repo and selected checkpoints are used to give the same values as reported in the paper. To change
  the configurations, check `config.py`.


- For Mogaze dataset

```bash
# For Prednet MSM
python experiments/mae_mogaze.py -a prednet -ag False

# For RED
python experiments/mae_mogaze.py -a red -ag False
```

Parameters: same as above. However, `--action`` parameter is not required.

#### For calculating VOE

- For HRI:

```bash
# run VOE on PredNet MSM  for all HRI scenarios
python experiments/voe_hri.py -a prednet --action combined --avoid_goal False

# run VOE on PredNet MSMwG  for all HRI scenarios
python experiments/voe_hri.py -a prednet --action combined --avoid_goal True

# run VOE on RED for all HRI scenarios
python experiments/voe_hri.py -a red --action combined --avoid_goal False
```

- For Mogaze dataset

```bash
# VOE of PredNet MSM
python experiments/voe_mogaze.py -a prednet --avoid_goal False

# VOE on PredNet MSMwG
python experiments/voe_mogaze.py -a prednet --avoid_goal True

# VOE on RED
python experiments/voe_mogaze.py -a red --avoid_goal False
```

Note: time for calculating VOE takes long.

# 4. Dependencies  <a name="dependencies"></a>

This repo uses/includes:

- a modified version of [robosuite](https://github.com/ARISE-Initiative/robosuite): to create the workplaces in mujoco,
- sample data (p1_1 and p2_1) downloaded from [Mogaze](https://github.com/humans-to-robots-motion/mogaze) and
- code from [RED](https://github.com/una-dinosauria/human-motion-prediction) adapted for evaluation against PredNet.
- HRI human model (.mjcf) is a modified version from https://github.com/mingfeisun/DeepMimic_mujoco
- Mogaze human model (.mjcf) is created from the official human-model (.urdf) provided
  by [Mogaze](https://github.com/humans-to-robots-motion/mogaze)
- For the self developed HRI synthetic dataset, the code from [@Jiacheng Yang](https://github.com/yjc765) is used, which
  he developed during his master thesis at University of Stuttgart. However, only the generated data is provided in this
  repo.

We do not intend to violate any license of the aforementioned libraries or dependencies used in this project, some of
which are also listed under `env_OS.yml`. If we violate your license(s), please let us know!

# 5. Cite  <a name="cite"></a>

If you use our code, please cite us:

```
@INPROCEEDINGS{elshamouty_and_pratheepkumar_2021, 
author = {El-Shamouty, Mohamed and Pratheepkumar, Anish}, 
title = {PredNet: a simple Human Motion Prediction Network for Human-Robot Interaction},
booktitle = {2021 26th {IEEE} International Conference on Emerging Technologies and Factory Automation (ETFA)},
year = {2021} 
}
```

# 6. License  <a name="license"></a>

__MIT__

# 7. Acknowledgement  <a name="ack"></a>

We thank [@Danilo Brajovic](https://github.com/danilobr94) for checking the code before publishing it.
