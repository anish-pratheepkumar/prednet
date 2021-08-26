from .. import mjcf_utils
from ..arenas import Arena
from ..mjcf_utils import xml_path_completion
from ..object_base import ObjectBase


class EmptyArena(Arena):
    """Empty workspace."""

    # TODO should define the size of the arena
    def __init__(self):
        super().__init__(xml_path_completion("empty_arena.xml"))
        self.floor = self.worldbody.find("./geom[@name='floor']")

    def set_floor_attribute(self, attribute, value):
        self.floor.set(attribute, mjcf_utils.array_to_string(value))

    def add_table(self,
                  pos,
                  size,
                  type="box",
                  material: str = "table_mat",
                  mass: str = "2000",
                  friction: str = "1 0.005 0.0001",
                  **kwargs) -> str:
        table_name = ObjectBase.get_unique_name("table")
        table_geom = mjcf_utils.new_geom(name=table_name,
                                         pos=pos,
                                         geom_type=type,
                                         size=size,
                                         material=material,
                                         mass=mass,
                                         friction=friction, **kwargs)
        self.worldbody.append(table_geom)
        return table_name
