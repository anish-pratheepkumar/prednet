from ..arenas import Arena
from ..base import MujocoXML
from ..mjcf_utils import new_joint, xml_path_completion
from ...utils.robot_utils import get_geom


class Task(MujocoXML):
    """
    Base class for creating MJCF model of a task.

    A task typically involves a robot interacting with objects in an arena
    (workshpace). The purpose of a task class is to generate a MJCF model
    of the task by combining the MJCF models of each component together and
    place them to the right positions. Object placement can be done by
    ad-hoc methods or placement samplers.
    """

    # TODO in all children tasks extending this class, you should create a pointer to all objects in the task xml
    #  , e.g. to table, obstacles, floor, etc..

    def __init__(self, xml_path=xml_path_completion("base.xml")):
        super().__init__(xml_path)
        self.arena = None
        self.mujoco_objects = None
        self.n_objects = None
        self.objects = []

    def merge_arena(self, mujoco_arena: Arena):
        """Adds arena model to the MJCF model."""
        self.arena = mujoco_arena
        self.merge(mujoco_arena)
        return self

    @property
    def floor(self):
        return get_geom(self.worldbody, 'floor')

    def add_object(self, objects: list, body):
        for obj_mjcf in objects:
            body.append(obj_mjcf)
            self.objects.append(obj_mjcf)
        return self

    def merge_objects(self, mujoco_objects: list, add_free_joint=False):
        """
        Adds physical objects to the MJCF model.
        :param mujoco_objects: list of objects
        :param add_free_joint: add free joint to enable the object to be manipulated/moved/ or fall
        :return:
        """
        self.n_objects = len(mujoco_objects)
        self.mujoco_objects = mujoco_objects
        # self.objects = []  # xml manifestation
        for obj_mjcf in mujoco_objects:
            self.merge_asset(obj_mjcf)
            # Load object
            try:
                obj = obj_mjcf.get_collision(name=obj_mjcf.object_id, site=True)
                self.objects.append(obj)
            except:
                self.objects.append(obj_mjcf)
            if add_free_joint:
                obj_body = obj_mjcf.worldbody.find(".//body")
                obj_body.append(new_joint(name=obj_mjcf.object_id, type="free", damping="0.0005"))
            self.merge(obj_mjcf)
        return self

    def place_objects(self, is_free):
        """Places objects randomly until no collisions or max iterations hit."""
        pass
