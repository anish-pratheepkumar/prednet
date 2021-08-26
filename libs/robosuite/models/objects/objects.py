import numpy as np

from libs.robosuite.models import assets_root
from libs.robosuite.utils.robot_utils import get_body
from ..mjcf_utils import xml_path_completion, array_to_string
from ..object_base import ObjectBase


class EmptyObject(ObjectBase):

    def __init__(self, name, fname="objects/empty.xml", **kwargs):
        super().__init__(fname=xml_path_completion(fname), object_name=name, **kwargs)
        self.bottom_offset = np.array([0., 0., 0.])


class Goal(ObjectBase):
    def __init__(self, fname="goal.xml", object_name="Goal"):
        super().__init__(xml_path_completion(fname, given_assets_root=assets_root), object_name=object_name,
                         add_base=False)
        self.bottom_offset = np.array([0., 0., 0.])
        self.set_base_xpos(np.zeros(3))

    def set_base_xpos(self, pos):
        node = get_body(self.worldbody, "Goal")
        node.set("pos", array_to_string(pos - self.bottom_offset))


if __name__ == '__main__':
    Goal()
