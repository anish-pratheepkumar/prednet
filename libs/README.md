# Robosuite
This repo is dedicated for modelling robotic workplaces, forked from [Surreal Robotics Suite v0.3.0](https://github.com/ARISE-Initiative/robosuite/tree/v0.3.0). With Robosuite, we define:
1. Modelling API - modelling robotic workplaces including Products, Resources (robots, grippers, humans, safety sensors) and processes (IK solver, Pitasc force-controller, etc. )
2. Simulation API - creating the modelled robotic workplaces in [MuJoCo physics engine](http://www.mujoco.org/book/index.html)
![c175cc61a6cb2fc84520a79d2678ba21.png](docs/_references/c2f55908377744bf84e42a3cf996099c.png)

Below we include the installation steps and tutorials to get started. Full documentation of the framework is under progress.

# Prerequisites
The framework is tested on Ubuntu (>=18.04), and the installation in this README is done using [Conda](https://docs.conda.io/en/latest/).

# Installation
To start using the repo, first clone using
1. Make sure to clone the repo with submodules:
    ```shell script
    git clone --recursive https://gitlab.cc-asp.fraunhofer.de/mae/robosuite.git 
    ```
2. Go to the project directory `robosuite`
3. Install the conda env using (currently for linux only)
    ```shell script
    conda env create -f env_linux.yml
    ```
	> Note: if you already have an existing conda env, you can extend it by calling `conda env update --name MY_ENV --file env_linux.yml`
# Conventions
**DO NOT** extend robosuite with additional models. Always extend on your own fork! Only bug fixes are allowed for now to 
be merged with `robosuites` to keep robosuites minimal and clean. Exception is made for adding new robot and gripper models (has to be confirmed with [@MAE](https://gitlab.cc-asp.fraunhofer.de/mae)).

# Tutorials
The tutorials depict main functionallities of robosuites. It is recommended to:
1. Do the tutorials in order (optional tutorials can be skipped)
2. Read the accompanying notes represented as 
   	```python
	# (1) code comments  
	```
 	and
	> (2) inline notes 
3. Check the tutorials full implementation under `docs/tutorials`

## Prerequisites
Before starting the tutorials:
1. create the folder `robosuite/tutorial`. Inside this folder, create directories or python files for the tutorials as mentioned in the respective tutorial.
2. Enable your conda environment 
3. Make sure that the package imports are correct (based on from where you open the project). If you use pycharm to open robosuite (as the main project), the imports below will work. Otherwise, you can always point to robosuite using:
```python
import sys
dir_path = "PATH_TO_ROBOSUITE_REPO"
sys.path.insert(0, dir_path)
```
## T0: Creating a WorkplaceEnv
This task introduces how to create an empty Workplace environment consisting of an empty arena (with just a floor and light properties).

Create the `robosuite/tutorial/t0/t0.py` and add the following code to creat the workplace environment:
```python
# Create an empty task with EmptyArena
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import Task

# Other tasks and arenas are under /robosuite/models/tasks/ and /robosuite/models/arenas/. 
task = Task().merge_arena(EmptyArena())

# Create the environment (dynamic model) from the task (static model)
from robosuite.environments.workplaces import WorkplaceEnv
workplace = WorkplaceEnv(task, has_renderer=True)

# saves the xml model of the workplace
workplace.model.save_model("workplace.xml")

# visualize the environment
while True:
	workplace.render()
```
> Note:
> - The `WorkplaceEnv` combines the static layout (Task + Arena + Robot Models), dynamic  components (any object) extending `MujocoInitEnv` (such as `RobotEnv`, `SensorEnv`, etc.) which are either Actionable or Observable. Actionable means: the object has `_pre_action`, `_set_action` and `_post_action`. Observable means the object has `_step_` and  `_get_observation_as_dict`.
> - In the constructor of `WorkplaceEnv, all the xml models from all objects in tasks, actionables and observables are suggregated and used to generate a single XML model, saved in `workplace.model`. This model is used to create the mujoco simulation instance `MJSim` (from `mujoco_py`). MJViewer is also created based on the parameter `has_renderer` that is passed to `WorkplaceEnv`.
> - Check the generated `workplace.xml` corresponding to the environment and run it with Mujoco's simulate command


## T1: Modelling a Product (3D assets in general)
The aim of this task is to model a 3D asset that can be handled by the robot.
> **Note**: 
> 1. Mujoco supports assets of type binary stl. If the robot assets are represented in other formats, it can be converted to binary stl using [vtk](https://pypi.org/project/vtk/) (already included in `env_linux.yml`). A sample code for converting ASCII STL to binary STL is under `robosuite.utils.mesh_tools.py`. 
> 2. [Mujoco works with triangulated meshes, and supports a maximum of 20K faces per STL file](http://www.mujoco.org/book/XMLreference.html#mesh)
> 3. [Mujoco calculates collisions using the convex hull of the objects](http://www.mujoco.org/book/computation.html#Collision). Each [geom asset is automatically approximated by a convex hull](http://www.mujoco.org/book/XMLreference.html#mesh). In many cases, the convex hull approximation removes a lot of details from the original asset (see figure below - left and middle shapes). Therefore, it is adviced that: if details are required from mesh assets, decompose these assets into a smaller set of convex hulls (figure below - right shape). A sample code for convex decomposition of assets is under `robosuite.utils.mesh_tools.py`.
> ![06841f9f4d724c07613058044c76118a.png](docs/_references/f498b9ba5ff940a5bff40b3fcf921c73.png) 
Image from [V-HACD](https://github.com/kmammou/v-hacd).

Steps:
1. Create the folders `robosuite/tutorial/t1/model/` and add in `model` `__init__.py`.
2. Create the folder `robosuite/tutorial/t1/model/assets/`
3. Add the following code to the `robosuite/tutorial/t1/model/__init__.py`:
	```python
	import os
	# The following line is to create a pointer to the absolute path of the assets folder in a clean way
	assets_root = os.path.join(os.path.dirname(__file__), "assets")
	```
4. Create `assets/meshes/` and copy the asset (binary stl) file `robosuite/models/assets/objects/meshes/milk.stl` to it
5. Create `assets/textures/` and copy the texture (.png) and `robosuite/models/assets/textures/ceramic.png` to it
6. Create `product.xml` in `assets/` and add the following code:
	```xml
	<mujoco model="product">
		<asset>
			<!-- Assets here are in meters! Asset should be re-scaled to meters if they are represented in the stl file in mm -->
			<mesh file="meshes/milk.stl" name="product_mesh"  scale="1. 1. 1."/>
			<!-- Texture is only for visualization. It needs an image as an input -->
			<texture file="textures/ceramic.png" name="tex-ceramic"/>
			<!-- Materials are created based on textures -->
			<material name="ceramic" reflectance="0.5" texrepeat="1 1" texture="tex-ceramic" texuniform="true"/>
		</asset>
		<worldbody>
				<body name="product">
					<geom name="product" pos="0 0 0" mesh="product_mesh" material="ceramic" type="mesh"/>
				</body>
		</worldbody>
	</mujoco>
	```
	> **Note:** Meshes in mujoco are assumed to be in meters! CAD softwares export meshes usually in mm. Therefore, in `<mesh ... scale="..."/>` consider changing scale to `scale="0.001 0.001 0.001"` to convert from mm to meters.
7. The folder structure should look like this:
	```bash
   t1
   ├── model
   │   ├── assets
   │   │   ├── meshes
   │   │   │   └── milk.stl
   │   │   ├── product.xml
   │   │   └── textures
   │   │       └── ceramic.png
   │   └── __init__.py
   ├── t1.py
   └── workplace.xml

	```
9. In `t1/model`, create `t1.py`, and add the following:
	```python
	# Necessary imports
	from robosuite.models.object_base import ObjectBase
	from robosuite.models.mjcf_utils import xml_path_completion, array_to_string
	from robosuite.utils.robot_utils import get_body
	import numpy as np

	# Create an empty task with EmptyArena
	from robosuite.models.arenas import EmptyArena
	from robosuite.models.tasks import Task

	task = Task().merge_arena(EmptyArena())

	# Create the product class
	from model import assets_root  # path to the assets folder
	class Product(ObjectBase):
		"""
		Class for instantiating python objects from product.xml
		"""

		def __init__(self, fname="product.xml", object_name="product"):
			super().__init__(xml_path_completion(fname, given_assets_root=assets_root), object_name=object_name)
			self.bottom_offset = np.array([0., 0., 0.])  # added to the body position automatically in set_base_xpos
			# self.bottom_offset = np.array([0., 0., -0.08])  # sets the milk to be above the floor

			# place the object body in the center w.r.t to the parent body (in our case the world)
			self.set_base_xpos(np.zeros(3))

		def set_base_xpos(self, pos):
			node = get_body(self.worldbody, "product")  # body name
			node.set("pos", array_to_string(pos - self.bottom_offset))


	# Create a product instance and add it to the task
	product_1 = Product()
	task.merge_objects([product_1])
	
	# 1. Uncomment this to rotate product_1
	# product_base_body = product_1.base_body
	# product_base_body.set("euler", "1.57 0 0")

	# 2. Uncomment this if you want to add another product
	# product_2 = Product()
	# product_2.set_base_xpos(np.array([0., -0.5, 0.]))
	# task.merge_objects([product_2])

	# Create the Workplace and visualize it
	from robosuite.environments.workplaces import WorkplaceEnv

	workplace = WorkplaceEnv(task, has_renderer=True)

	# saves the xml model of the workplace
	workplace.model.save_model("workplace.xml")

	# visualize the environment
	while True:
		workplace.render()

	```
> **Note**:
> 1. Product extends `ObjectBase > MujocoXML`. You can access all the xml elements of the product, and add/modify/remove attributes. Example, try rotating the milk product by getting its base body and rotating it
> 2. Try to add more milk products and align them beside each other
> - Check the generated `workplace.xml` corresponding to the environment and run it with Mujoco's simulate command



## T2: Adding a Robot to a workplace and moving it
In this tutorial, we will add multiple robots to the workplace and control them using a Mocap (inverse kinematics) and joints.
Create `robosuite/tutorial/t2/t2.py` and add the following
```python
import numpy as np
from robosuite.environments.robots.base import RobotEnv
from robosuite.environments.workplaces import WorkplaceEnv
from robosuite.models.robots import Ur5
from robosuite.models.tasks import ReachTask

# Create the task
task = ReachTask()  # a task with a table
# Create an instance of the robot static model
# 0. Supported robot models are found under `robosuite/models/robots/__init__.py`
# 1. Uncomment one of the next lines to control the robot using joints or mocap (inverse kinematics)
robot_model = Ur5(control_using_mocap_not_joints=True)  # control the robot using Mocaps
# robot_model = Ur5(control_using_mocap_not_joints=False)  # control the robot using joints

robot_model.set_base_xpos(pos=[-0.25, -0.25, 0.4])  # place the UR5 on the table

# Create an instance of the robot dynamic model (to enable moving the robot and calculating dynamics)

# 2. Uncomment one of the next lines to control the robot using absolute actions or delta actions, and set the robot
# control frequency
robot_env = RobotEnv(model=robot_model, is_robot_action_delta=False, control_freq=-1)  # absolute actions
# robot_env = RobotEnv(model=robot_model, is_robot_action_delta=True, control_freq=-1) # actions are relative
# postions (to gripper) instead of absolute positions

# 3. Play with the robot frequency above and check the corresponding behaviour, e.g., try  control_freq=1 and
# control_freq=125

# Create the workplace Env containing the task, and the robot_env as an actionable
# 4. n_substeps define for each sim.step(), how many simulation substeps are executed. Try out different n_substeps and
# check the behaviour of the robot
workplace_env = WorkplaceEnv(actionables=[robot_env], task=task, has_renderer=True, n_substeps=20)
workplace_env.model.save_model("workplace.xml")

while True:
	# 5. Uncomment one of the next lines to control the robot differently
	action = robot_env.grip_pos + np.random.uniform(-0.05, 0.05, 3)  # move gripper to a given position
	# action = robot_env.grip_orientation + np.random.uniform(-0.05,0.05,4) # rotate gripper to a random rotation
	# action = robot_env.grip_pose + np.random.uniform(-0.05,0.05, 7) # move and rotate gripper to a given pose
	# action = np.concatenate((robot_env.grip_pose + np.random.uniform(-0.05, 0.05, 7), [
	#     np.random.choice([0, 1])]))  # move and rotate gripper to a given pose and Open/Close gripper

	# passes the action to the actionable (robot_env) of the workplace env and steps the simulation
	workplace_env.step(action)
	workplace_env.render()
```
This will create the following workplace. 
![b92bde54eebeb1a56fa85e08490888ff.png](docs/_references/673280a413c6411a91e3277be6f676f1.png)

> **Note**: 
> 0. Supported robot models are found under `robosuite/models/robots/__init__.py` and supported Gripper models are found under `robosuite/models/grippers/gripper_factory.py`
> 1. Change `control_using_mocap_not_joints` to control the robot using **joints or mocaps**
> 2. Control the robot's gripper position using **relative or delta position**: when initializing `robot_env`, set `is_robot_action_delta=True` 
> 3. Set the **robot's control  frequency**. Default value = -1, which means the robot has the same control frequency as the frequency of the simulation (1/(sim.model.opt.timestep*self.sim.nsubsteps))
> 4. **Simulation granularity:** `n_substeps=20` sets the simulation substeps to be 20, meaning that we pass a single action to the simulation, and then we calculate 20 substeps inside the simulation (includes 20 IK iterations to reach the given positions - [Check this reference in mujoco-py docs](https://openai.github.io/mujoco-py/build/html/reference.html#mujoco_py.MjSim)). Setting this variable to a small value (e.g., 1) can cause the IK solver to be not accurate in reaching the position, and setting it too high (e.g., 100) can slow down the simulation and reduce the responsiveness of the robot (i.e. the robot's control frequency).
> 5. **Robot Action**:
> 	 - Control the robot's gripper orientation: pass an action as quaternions of size 4 (qw, qx, qy, qz): `action = robot_env.grip_orientation + np.random.uniform(4)`. If euler angles are prefered, than refer to `gym.envs.robotics.rotations.euler2quat()`
> 	 - Control the robot's pose (position + orientation): pass an action of size 7 (x, y, z, qw, qx, qy, qz): `action = robot_env.grip_pose + np.random.uniform(7)`
> 	 - Control the robot's pose and gripper. Note that you pass to the gripper {0: open | 1: close} at each timestep, and this will be translated automatically to the corresponding gripper joint angles (refer to `robosuite.environments.robots.base._set_action()`)


## T3: Attach a force-torque sensor to the robot
1. Create `robosuite/tutorial/t3/t3.py` and add the following to create the workplace including a UR5 with a force-torque sensor:
	```python
	import numpy as np

	from robosuite.environments.robots.base import RobotEnv
	from robosuite.environments.workplaces import WorkplaceEnv
	from robosuite.models.robots import Ur5
	from robosuite.models.tasks import ReachTask

	task = ReachTask()
	robot_model = Ur5(control_using_mocap_not_joints=True)
	robot_model.set_base_xpos(pos=[-0.25, -0.25, 0.4]) 

	# Attach the force-torque sensor to the robot by setting
	robot_env = RobotEnv(model=robot_model, is_robot_action_delta=False, add_force_torque_sensor=True)  # absolute actions
	workplace_env = WorkplaceEnv(actionables=[robot_env], task=task, has_renderer=True, n_substeps=20)
	workplace_env.model.save_model("workplace.xml")

	total_steps = 100
	force_readings = np.zeros((total_steps, 3))
	torque_readings = np.zeros((total_steps, 3))
	for i_step in range(total_steps):
		action = robot_env.grip_pos + np.array([0., 0., -0.05])  # move gripper downwards to hit the table
		workplace_env.step(action)
		force_readings[i_step, :] = robot_env.force_reading
		torque_readings[i_step, :] = robot_env.torque_reading
		workplace_env.render()
	```
2. Plot the force and torque readings
	```python
	# plot force reading
	labels_force = ['f_x', 'f_y', 'f_z', ]
	labels_torque = ['t_x', 't_y', 't_z', ]
	import matplotlib.pyplot as plt

	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	ax1.title.set_text('Force Readings')
	ax2.title.set_text('Torque Readings')
	x_axis_range = np.arange(0, total_steps)
	for i_axis in range(3):
		ax1.plot(x_axis_range, force_readings[:, i_axis], label=labels_force[i_axis])
		ax1.legend(loc="upper right")
		ax2.plot(x_axis_range, torque_readings[:, i_axis], label=labels_torque[i_axis])
		ax2.legend(loc="upper right")
	plt.show()
	```
> Note:
> - You can observe that the force readings are too high (more than normal) after the robot hits the table. Factors affecting the accuracy of the forces calculation will be added in the future.
> - Torque readings are affected by joint dampings used in the xml file. Play with this parameter in `robosuite/models/assets/robots/ur5/ur5.xml > line 33` default parameter for joints and observe how the torque readings differ

   
# Extra Tutorials (Optional)
## TE1: Modelling a Robot 
Modelling robots in Mujoco can be done in multiple ways, namely, (1) using existing .URDF file, and (2) using .mjcf file. In this tutorial, we model a UR10e robot using a pre-existing URDF and asset files (i.e., method 1). This will help you understand how robots are modelled in robosuites.

1. Fetch the URDF and asset files corresponding to the desired robot (in this case UR10e, available under **XXXXX**)
> **Note**:
> 1. URDF and URDF.XACRO: nowadays, robots are usually modelled using `urdf.xacro` instead of `.urdf`. [Mujoco supports only `.urdf`](http://mujoco.org/book/modeling.html#CURDF). You can [convert from `.urdf.xacro` to `-urdf`](https://answers.ros.org/question/10401/how-to-convert-xacro-file-to-urdf-file/) using the command `rosrun xacro xacro --inorder -o model.urdf model.urdf.xacro`.
> 2. Assets have to be referenced correctly in the URDF. 
> 3. MJCF: empty [`default` classes](http://mujoco.org/book/modeling.html#CDefault) are not supported! `default` should always have a name other than `main`.


2. Open the .urdf using mujoco simulate
3. Save the corresponding .mjcf generated by Mujoco by clicking on `File > Save xml`. 
4. Rename the generated xml to `ROBOT_NAME.xml`
5. Create XML model under `robosuite/models/assets/robots`, with the following structure:
	
	```bash
	ROBOT_NAME
	├── meshes
	│   ├── # place here the stl files of the robot (binary .stl)
	├── textures
	│   └── # place here the texture files of the robot (.png)
	└── ROBOT_NAME.xml
	```
	`ROBOT_NAME` should be replaced by the respective robot name.

6. Create the corresponding Python wrapper `robosuite/models/robots/ROBOT_NAME.py`. The robot should extend `robosuite.models.robots.robot`. Note that you will need to extend the functions in `robot`.
7. Define the added robot python wrapper under `robosuite.models.robots.__init__.py` in the following format:
	```python
	from .ROBOT_NAME import ROBOT_NAME
	```
8. Create an instance from the new robot model, and pass it to `RobotEnv` to have a dynamic model, similar to:
	```python
	from robosuite.models.robots import Ur5 # here we use Ur5 robot model
	robot_model = Ur5(control_using_mocap_not_joints=True)
	robot_env = RobotEnv(model=robot_model, is_robot_action_delta=False)
	
	# Create the 'workplaceEnv normally
	workplace_env = WorkplaceEnv(actionables=[robot_env], task=task, has_renderer=True, n_substeps=20)
	while True:
		workplace_env.render()
	```


## TE2: Create a Learning Environment

There are two ways to create a learning workplace:
1. `LearnWorkplace`: extends `WorkplaceEnv` directly, which means after training, the policy is only possible to run in 
this LearnWorkplace without possible combintation with other policies (e.g., learnt policy for human walking cannot 
be combined with policy for human lifting boxes)
2. `SubWorkplaceEnv` (still experimental): train the policy in this workplaceenv, and later combine it with other policies
from other `SubWorkplaceEnv`, i.e., 3 policies can be combined in one Workplace to control the same model (e.g., human 
walk + list + wipe, check test_hrc.py for concrete example)

### a. Creating `LearnWorkplace` or `SubWorkplaceEnv`
Both `LearnWorkplace` and `SubWorkplaceEnv` have similar structures. To create any of them, you have to:
1. create your own `LearnWorkplace` or `SubWorkplaceEnv`, then extend one of the two classes
2. Override the environment specific methods (inherited from `MujocoinitEnv`):
    ```python
        def _load_model(self):
            """
            (Optional) Do here any modifications to the model before initializing the sim
            """
            super()._load_model()   
            pass
    
        def _reset_internal(self):
            """
            resets the environment
            """
            raise NotImplementedError()
    
        def _get_reference(self):
            """
            Called after sim is initialized. Make the necessary bindings from self.model (the XML) to sim
            """   
            raise NotImplementedError()
    
        def _get_observation_as_dict(self):
            """ Return observations (if any) in the form of OrderedDict({...})"""   
            raise NotImplementedError()
    
        def step(self, action=None):
            """
            Called each timestep (before _set_action is called if self is `Actionable`). Can be used for stepping 
            the objects, e.g. moving obstacles, moving goals
            """
            pass
    ```
3. Override the learning specific methods (inherited from `LearnWorkplaceEnv`):
    ```python
    def is_success(self, observations, desired_goal, achieved_goal):
        raise NotImplementedError()

    def compute_reward(self, observations, desired_goal, achieved_goal, info={}):
       raise NotImplementedError()
    ```
   **Note**: you can override as well `_get_observation_as_dict` if you want to combine observations based on multiple
   observables (e.g., assume two observables: goal_env and human_env, we need to add as an observation the relative distance
   between the goal and the human)

After creating the LearnWorkplace, you can now create your Observables and Actionables.
### b. Creating the observables
The observables should extend `MujocoInitEnv` which requires properly passing the XML model to the observable in the 
constructor and extending the functions in `MujocoInitEnv` as in previous paragraph step 2.
### c. Creating the actionables
1. The actionables are exactly the same as the observables (follow the previous step).
2. Still, you need to further implement
`_pre_action`, `_set_action` and `_post_action` which defines what kind of action is supported by your actionable.
3. make sure you define the degrees of freedom (DoF) proporly in your xml model that you pass to the actionable. This specifies
the action_specs of the actinoale. An alternative is to override the following method:
   ```python
    @property
    def action_spec(self):
        """
        Action lower/upper limits per dimension. This depends on the model.dof
        """
        low = np.ones(self.model.dof) * -1. 
        high = np.ones(self.model.dof) * 1.
        return low, high
    ```
	
## TE3: Exceptions Handling
### (a) Debugging model exceptions (errors in the generated XML)
If the XML of one of the resources/products is wrong, the overall XML will be generated and saved under the default path `os.getcwd()+'/EXCEPTION.xml'`. To understand the error, you need to simulate the generated `EXCEPTION.xml`using Mujoco simulate command, and debug their the error.
## TE4: Adding a depth_camera to the env
- [ ] To be added

## TE5: Robot Grasping and object
- [ ] To be added


## TE6: Creating a hard-coded Robotic Workplace "the dirty-way"

1. Create the desired xml file representing the workplace. The `Arena` describes the static environment or static components in a robot cell layout. 
   The task current encapsulates the `Arena` (to define how to sample starting positions of the static objects) but will be removed in a future version.
    ```xml
    <mujoco model="planar manipulator">

    <asset>
        <texture name="background" builtin="flat" type="2d" mark="random" markrgb="1 1 1" width="800" height="800"
                 rgb1=".2 .3 .4"/>
        <material name="background" texture="background" texrepeat="1 1" texuniform="true"/>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"
                 mark="edge" markrgb=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
        <material name="self" rgba=".7 .5 .3 1"/>
        <material name="self_default" rgba=".7 .5 .3 1"/>
        <material name="self_highlight" rgba="0 .5 .3 1"/>
        <material name="effector" rgba=".7 .4 .2 1"/>
        <material name="effector_default" rgba=".7 .4 .2 1"/>
        <material name="effector_highlight" rgba="0 .5 .3 1"/>
        <material name="decoration" rgba=".3 .5 .7 1"/>
        <material name="eye" rgba="0 .2 1 1"/>
        <material name="target" rgba=".6 .3 .3 1"/>
        <material name="target_default" rgba=".6 .3 .3 1"/>
        <material name="target_highlight" rgba=".6 .3 .3 .4"/>
        <material name="site" rgba=".5 .5 .5 .3"/>
        <texture name="skybox" type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2="0 0 0"
                 width="800" height="800" mark="random" markrgb="1 1 1"/>
    </asset>

    <visual>
        <map shadowclip=".5" znear=".01"/>
        <quality shadowsize="2048"/>
        <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
    </visual>
    >

    <option timestep="0.001" cone="elliptic"/>

    <default>
        <geom friction=".7" solimp="0.9 0.97 0.001" solref=".005 1"/>
        <joint solimplimit="0 0.99 0.01" solreflimit=".005 1"/>
        <general ctrllimited="true"/>
        <tendon width="0.01"/>
        <site size=".003 .003 .003" material="site" group="3"/>

        <default class="arm">
            <geom type="capsule" material="self" density="500"/>
            <joint type="hinge" pos="0 0 0" axis="0 -1 0" limited="true"/>
            <default class="hand">
                <joint damping=".5" range="-10 60"/>
                <geom size=".008"/>
                <site type="box" size=".018 .005 .005" pos=".022 0 -.002" euler="0 15 0" group="4"/>
                <default class="fingertip">
                    <geom type="sphere" size=".008" material="effector"/>
                    <joint damping=".01" stiffness=".01" range="-40 20"/>
                    <site size=".012 .005 .008" pos=".003 0 .003" group="4" euler="0 0 0"/>
                </default>
            </default>
        </default>

        <default class="object">
            <geom material="self"/>
        </default>

        <default class="task">
            <site rgba="0 0 0 0"/>
        </default>

        <default class="obstacle">
            <geom material="decoration" friction="0"/>
        </default>

        <default class="ghost">
            <geom material="target" contype="0" conaffinity="0"/>
        </default>
    </default>

    <worldbody>
        <!-- Arena -->
        <light name="light" directional="true" diffuse=".6 .6 .6" pos="0 0 1" specular=".3 .3 .3"/>
        <geom name="floor" type="plane" pos="0 0 0" size=".4 .2 10" material="grid"/>
        <geom name="wall1" type="plane" pos="-.682843 0 .282843" size=".4 .2 10" material="grid" zaxis="1 0 1"/>
        <geom name="wall2" type="plane" pos=".682843 0 .282843" size=".4 .2 10" material="grid" zaxis="-1 0 1"/>
        <geom name="background" type="plane" pos="0 .2 .5" size="1 .5 10" material="background" zaxis="0 -1 0"/>
        <camera name="fixed" pos="0 -16 .4" xyaxes="1 0 0 0 0 1" fovy="4"/>

        <!-- Arm -->
        <geom name="arm_root" type="cylinder" fromto="0 -.022 .4 0 .022 .4" size=".024" material="decoration"
              contype="0" conaffinity="0"/>
        <body name="upper_arm" pos="0 0 .4" childclass="arm">
            <joint name="arm_root" damping="2" limited="false"/>
            <geom name="upper_arm" size=".02" fromto="0 0 0 0 0 .18"/>
            <body name="middle_arm" pos="0 0 .18" childclass="arm">
                <joint name="arm_shoulder" damping="1.5" range="-160 160"/>
                <geom name="middle_arm" size=".017" fromto="0 0 0 0 0 .15"/>
                <body name="lower_arm" pos="0 0 .15">
                    <joint name="arm_elbow" damping="1" range="-160 160"/>
                    <geom name="lower_arm" size=".014" fromto="0 0 0 0 0 .12"/>
                    <body name="hand" pos="0 0 .12">

                        <joint name="arm_wrist" damping=".5" range="-140 140"/>
                        <geom name="hand" size=".011" fromto="0 0 0 0 0 .03"/>
                        <site name="grasp" pos="0 0 .065"/>
                        <site name="palm_touch" type="box" group="4" size=".025 .005 .008" pos="0 0 .043"/>
                    </body>
                </body>
            </body>
        </body>

        <!-- props -->
        <body name="peg" pos="-.4 0 .4" childclass="object">
            <joint name="peg_x" type="slide" axis="1 0 0" ref="-.4"/>
            <joint name="peg_z" type="slide" axis="0 0 1" ref=".4"/>
            <joint name="peg_y" type="hinge" axis="0 1 0"/>
            <geom name="blade" type="capsule" size=".005" fromto="0 0 -.013 0 0 -.113"/>
            <geom name="guard" type="capsule" size=".005" fromto="-.017 0 -.043 .017 0 -.043"/>
            <body name="pommel" pos="0 0 -.013">
                <geom name="pommel" type="sphere" size=".009"/>
            </body>
            <site name="peg" type="box" pos="0 0 -.063"/>
            <site name="peg_pinch" type="box" pos="0 0 -.025"/>
            <site name="peg_grasp" type="box" pos="0 0 0"/>
            <site name="peg_tip" type="box" pos="0 0 -.113"/>
        </body>

        <!-- receptacles -->
        <body name="slot" pos="-.405 0 .2" euler="0 20 0" childclass="obstacle">
            <geom name="slot_0" type="box" pos="-.0252 0 -.083" size=".0198 .01 .035"/>
            <geom name="slot_1" type="box" pos=" .0252 0 -.083" size=".0198 .01 .035"/>
            <geom name="slot_2" type="box" pos="  0   0 -.138" size=".045 .01 .02"/>
            <site name="slot" type="box" pos="0 0 0"/>
            <site name="slot_end" type="box" pos="0 0 -.05"/>
        </body>


        <!-- targets -->
    </worldbody>

    <equality>
        <!-- <tendon name="coupling" tendon1="coupling" solimp="0.95 0.99 0.001" solref=".005 .5"/> -->
    </equality>

    <sensor>
        <force name="palm_force" site="palm_touch"/>
        <touch name="palm_touch" site="palm_touch"/>
    </sensor>

    <actuator>
        <motor name="root" joint="arm_root" ctrlrange="-1 1" gear="12"/>
        <motor name="shoulder" joint="arm_shoulder" ctrlrange="-1 1" gear="8"/>
        <motor name="elbow" joint="arm_elbow" ctrlrange="-1 1" gear="4"/>
        <motor name="wrist" joint="arm_wrist" ctrlrange="-1 1" gear="2"/>
    </actuator>

    </mujoco>
    ```
2. Create a task from your xml file (with `xml_path`)
    ```python
    class PegTask(Task):
    """Static Model"""
       def __init__(self, xml_path):
           super().__init__(xml_path))
    ```
3. Create the corresponding workplace
    ```python
    from collections import OrderedDict
    import numpy as np
    from academy.workplaces.base import LearnWorkplaceEnv
    from academy.workplaces.control_problem.models import assets_root
    from robosuite.robosuite.models.mjcf_utils import xml_path_completion
    from robosuite.robosuite.models.tasks import Task
    from robosuite.robosuite.utils import XMLError
    from robosuite.robosuite.utils.mocap_utils import ctrl_set_action_with_actuator_ref
    from robosuite.robosuite.utils.robot_utils import get_sensor, get_body
    
    class ManipulatorWorkplace(LearnWorkplaceEnv):
        """ Dynamic model and learning environment """
        def _reset_internal(self):
            super(ManipulatorWorkplace, self)._reset_internal()
    
        def _load_model(self):
            super(ManipulatorWorkplace, self)._load_model()
    
        def _get_reference(self):
            """ Get IDs of objects to be referenced from the sim (connection between static model and dynamic model"""
            # Access to sensors
            palm_touch_sensor_model = get_sensor(self.model.sensor, sensor_type="touch", name="palm_touch")
            self.palm_touch = self.sim.model.sensor_name2id(palm_touch_sensor_model.get("name"))
            self.sensors = {sensor.get("name"): self.sim.model.sensor_name2id(sensor.get("name")) for sensor in
                            self.model.sensor}
            # Access to motors
           self.motor = {motor.get("name"): self.sim.model.actuator_name2id(motor.get("name")) for motor in
                          self.model.actuator.findall(".//motor")}
            # Access to peg body
            self.peg_body = get_body(self.model.worldbody, "peg")
            self.peg_body_id = self.sim.model.body_name2id(self.peg_body.get("name"))
    
            # Access to motors
            pass
    
        def _get_observation_as_dict(self):
            di = OrderedDict()
            di.update({sensor_name: self.sim.data.sensordata[sensor_id] for sensor_name, sensor_id in self.sensors.items()})
            di['peg_pos'] = self.sim.data.body_xpos[self.peg_body_id]
            return di
    
        def _pre_action(self, action):
            self._set_action(action)
    
        def _set_action(self, action):
            ctrl_set_action_with_actuator_ref(self.sim, list(self.motor.values()), action, False)
    
        def _post_action(self, obs, goal, achieved_goal, info=None):
            return None
    ```
Accordingly, the workplace environment contains the actionables (with `_pre_action`, `_set_action`
 and `_post_action`) and observables (with `_get_observation_as_dict()`). 

Example for running the above environment:
```python
if __name__ == '__main__':
    task = PegTask()
    # task.save_model(task.xml") # saves the task model
    workplace = ManipulatorWorkplace(task=task, has_renderer=True, dof=4)
    #workplace.model.save_model(workplace.xml") # saves the workplace model
    workplace.reset()
    while True:
        obs, reward, done, info = workplace.step(action=np.random.uniform(-1, 1, size=(4,)))
        # obs_as_dict = workplace._get_observation_as_dict() # to get human readable observation (dict) 
        print(obs, reward, done, info)
        workplace.render()

```