from ..base import MujocoXML
from ..mjcf_utils import xml_path_completion


class WorkplaceModel(MujocoXML):
    """
    Base class for creating MJCF model of a task.

    A task typically involves a robot interacting with objects in an arena
    (workshpace). The purpose of a task class is to generate a MJCF model
    of the task by combining the MJCF models of each component together and
    place them to the right positions. Object placement can be done by
    ad-hoc methods or placement samplers.
    """

    # TODO set dict with all things as a start here to get references later from the simulation?

    def __init__(self, robots: list = None, task=None):
        super().__init__(xml_path_completion("base.xml"))
        # self.robots = []
        # TODO Manage here xml options, compiler requirements, etc.
        self.task = None
        self.merge_task(task)
        for robot in set(robots):  # add unique robots
            self.merge_robot(robot)

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        # self.robots.append(mujoco_robot)
        self.merge(mujoco_robot)

    def merge_task(self, task):
        """Adds task to the workplace"""
        self.task = task
        self.merge(task)
