import copy
from abc import ABC
from collections import Iterable

import numpy as np
from gym.envs.robotics.rotations import quat2euler

from .mjcf_utils import new_element, array_to_string
from ..models import mjcf_utils
from ..models.base import MujocoXML
from ..utils.robot_utils import get_body, get_site, find_element_with_property, get_default_class

equality_elements = ['weld', 'connect']  # according to mjcf format


class ObjectBase(MujocoXML, ABC):
    """Base class for all robot models."""
    id = 0
    object_ids_dict = {}  # TODO make it unique per object type (e.g. a separate dict for site, geom, ... )

    def __init__(self, fname, object_name, control=False, control_using_mocap_not_joints=False, add_base=True, **_):
        """

        :param fname (str): filepath where the corresponding xml file is stored
        :param object_name (str): name of the object
        :param control (bool): flag if this object needs to be controlled
        :param with_mocap_not_joints_control (bool): true if the object is controlled with mocap, false if it is controlled with joints
        """
        global object_ids_dict
        self._dof = None
        super().__init__(fname)
        # ---- add id to objects with the same name
        self.object_name = object_name
        try:
            existing_robot = ObjectBase.object_ids_dict[object_name]
            self.object_id = object_name + str(existing_robot)
            ObjectBase.object_ids_dict[object_name] += 1
        except KeyError:
            self.object_id = object_name  # + str(0)
            # first object always have the exact assigned name so that we dont change the name if there is only one robot
            ObjectBase.object_ids_dict[object_name] = 1
        # -------------------------------------------
        self._add_base = add_base
        self.control = control
        self.control_using_mocap_not_joints = control_using_mocap_not_joints

        self.all_bodies_names_ = None
        self.all_geoms_names_ = None
        self.joint_names = None
        self.joint_limits_ = None
        self.n_joints = None  # this will invoke to get the joints before adding grippers!
        self.all_bodies_names_ = None
        self.all_geoms_names_ = None
        self.mocap_body = None
        self.init_object_base(False)
        # Remove contacts between direct bodies
        # self.remove_contact(self.base_body)
        # TODO instead of excluding contacts, apply contype and conaffinity

    def init_object_base(self, init_base=True):
        if init_base:
            self.initialize_base()

        if self._add_base:
            self.add_base()

        # if control: # TODO remove, to be called in manually after adding the gripper
        #   self.add_control(control_using_mocap_not_joints)

        self.rename_all()

        self.mocap_body = find_element_with_property(self.worldbody,
                                                     element_tag="body",
                                                     property_tag="mocap",
                                                     property_prefix="true")

    @property
    def base_body(self):
        """Places the object on position @pos."""
        base_body = find_element_with_property(self.worldbody, "body", "name", ":base", False)
        try:
            name = base_body.get("name")
        except:
            # select first body in the robot model
            base_body = self.worldbody.find("./body")
            try:
                name = base_body.get("name")
            except:
                base_body = None
        return base_body

    def add_control(self, visualize_mocap=0, mocap_name=None):
        global object_ids_dict
        mocap_name = "mocap" if mocap_name is None else mocap_name
        if self.control and self.control_using_mocap_not_joints:
            mocap_name = ObjectBase.get_unique_name(mocap_name)
            existing_mocap = get_body(self.worldbody, mocap_name)
            if existing_mocap is not None:
                if isinstance(existing_mocap, list) and len(existing_mocap) == 0:
                    pass
                else:
                    return None
                # print(mocap_body)
            body = mjcf_utils.new_body(mocap_name, pos=[0, 0., 0.7], mocap="true")
            site_1 = mjcf_utils.new_site(type="box", size=[0.025, 0.025, 0.025], rgba=[1, 1, 0, visualize_mocap])
            site_2 = mjcf_utils.new_site(type="box", size=[0.005, 0.005, 0.005], rgba=[0.5, 0, 0, visualize_mocap])
            site_3 = mjcf_utils.new_site(type="box", size=[0.005, 0.005, 0.005], rgba=[0, 0.5, 0, visualize_mocap])
            site_4 = mjcf_utils.new_site(type="box", size=[0.005, 0.005, 0.005], rgba=[0, 0, 0.5, visualize_mocap])
            # add orientation visualizer
            """
            <site name="targetx" rgba="0 0 1 1" size="0.005 0.04" pos="0. 0. 0.04" type="cylinder"/>
            <site name="targety" rgba="1 0 0 1" size="0.005 0.04" pos="0.04 0 0" euler="0 1.57 0" type="cylinder"/>
            <site name="targetz" rgba="0 1 0 1" size="0.005 0.04" pos="0 0.04 0" euler="1.57 0 0" type="cylinder"/>
            """
            site_x = new_element("site", rgba="0 0 1 1", size="0.005 0.04", pos="0. 0. 0.04", type="cylinder")
            site_y = new_element("site", rgba="1 0 0 1", size="0.005 0.04", pos="0.04 0 0", euler="0 1.57 0",
                                 type="cylinder")
            site_z = new_element("site", rgba="0 1 0 1", size="0.005 0.04", pos="0 0.04 0", euler="1.57 0 0",
                                 type="cylinder")
            body.extend([site_x, site_y, site_z])
            body.append(site_1)
            body.append(site_2)
            body.append(site_3)
            body.append(site_4)
            self.worldbody.append(body)

            # self.modify_property_helper(self.worldbody, "joint", {"damping": "1e0"})
            self.mocap_body = find_element_with_property(self.worldbody,
                                                         element_tag="body",
                                                         property_tag="mocap",
                                                         property_prefix="true")
        elif self.control:
            for joint_name, joint in zip(self.joints, self.joints_elements):
                if "gripper" not in joint_name:
                    # get control range from joint
                    ctrlrange = joint.get("range") or "-3.14 3.14"
                    act = mjcf_utils.new_actuator(joint=joint_name, act_type="position", ctrlrange=ctrlrange,
                                                  ctrllimited="true", kp="10000", name=joint_name)
                    self.actuator.append(act)
                    self.modify_property_helper(self.worldbody, "joint", {"damping": "1e5"})
        return mocap_name

    def add_actuator(self, joint_names, actuator_type="position", **act_args):
        for joint_name in joint_names:
            act = mjcf_utils.new_actuator(joint=joint_name, act_type=actuator_type, name=joint_name, **act_args)
            self.actuator.append(act)

    def add_base(self):
        """
        TODO: this method should move everythin in worldbody to base_body if it adds a base body. Currently, it moves only the first body and geom
        """
        global id

        first_body = self.worldbody.find(".//body")
        first_geom = self.worldbody.find(".//geom")
        if first_body is None:
            pass  # add base body
        elif "base" in first_body.get("name"):
            return
        # No base link existing:
        base_body = mjcf_utils.new_body(name="base")

        if first_body is not None:
            # move first body to new body
            self.worldbody.remove(first_body)
            base_body.append(first_body)
        if first_geom is not None:
            try:
                self.worldbody.remove(first_geom)
                base_body.append(copy.deepcopy(first_geom))
            except ValueError as e:  # if geom not part of worldbody
                pass  # print(e)
        self.worldbody.append(base_body)
        return base_body

    def rename_all(self):
        # TODO add a default name to all bodied, geoms, sites!
        self.rename_helper("body")
        self.rename_helper("geom")
        self.rename_helper("site")
        self.rename_helper("joint")
        self.rename_helper("camera")
        # To force updating joint names
        self.joint_names = None

        # Appends object_name as a prefix to default.class
        self.rename_helper_other_tags("default", "class", self.default)
        # add child class to base_body later below
        self.rename_helper_other_tags("weld", "body1", self.equality)
        self.rename_helper_other_tags("weld", "body2", self.equality)
        self.rename_helper_other_tags("geom", "class", self.worldbody)
        self.rename_helper_other_tags("position", "class", self.actuator)
        # Appends object_name as a prefix to bodies pointing to default.class
        self.rename_helper_other_tags("body", "childclass", self.worldbody)
        #         <framequat name="site_sensor" objname="sensor_site" objtype="site"/>
        self.rename_helper_other_tags("framequat", "objname", self.sensor)

        # ObjectBase.search_and_rename("exclude", "body1", self.actuator, renamed_element_tag="body",
        #                              renamed_property_tag="name", renamed_parent=self.worldbody)

        if len(self.contact) > 0:
            ObjectBase.search_and_rename("exclude", "body1", self.contact, renamed_element_tag="body",
                                         renamed_property_tag="name", renamed_parent=self.worldbody)
            ObjectBase.search_and_rename("exclude", "body2", self.contact, renamed_element_tag="body",
                                         renamed_property_tag="name", renamed_parent=self.worldbody)

        if len(self.equality) > 0:
            for child_element in equality_elements:
                ObjectBase.search_and_rename(child_element, "body1", self.equality, renamed_element_tag="body",
                                             renamed_property_tag="name", renamed_parent=self.worldbody)
                ObjectBase.search_and_rename(child_element, "body2", self.equality, renamed_element_tag="body",
                                             renamed_property_tag="name", renamed_parent=self.worldbody)

        # change names and take care of pointers:
        # exclude contact --> body
        # actuator --> joint
        # sensor --> site
        actuator_types = ["position", "velocity", "motor", "cylinder"]
        for actuator_type_ in actuator_types:
            self.rename_helper_other_tags(actuator_type_, "joint", self.actuator)
            self.rename_helper_other_tags(actuator_type_, "name", self.actuator)

        sensor_types = ["force", "torque"]
        for sensor_type_ in sensor_types:
            self.rename_helper_other_tags(sensor_type_, "site", self.sensor)
            self.rename_helper_other_tags(sensor_type_, "name", self.sensor)

        self.rename_helper("mesh", inside_worldbody=False)  # rename meshes to avoid conflicting mesh names
        self.rename_helper_other_tags("geom", "mesh", self.worldbody)  # rename meshes to avoid conflicting mesh names
        # self.worldbody.findall(".//mesh")

        default_class = get_default_class(self.default, self.object_id)
        # TODO any default class without a name should be given a default name, and applied on all relevant elements
        #  without a default childclass or clas in the xml
        if default_class is not None and len(default_class) > 0:
            self.base_body.set("childclass", default_class.get("class"))

    def rename_helper(self, tag, inside_worldbody=True):
        # TODO make unique names per type, and do not rename if object name is already unique!!!
        if inside_worldbody:
            objects = self.worldbody.findall(".//" + tag)
        else:
            objects = self.asset.findall(".//" + tag)

        id_for_objects_with_no_name = 0
        for o in objects:
            name = o.get("name") or str(id_for_objects_with_no_name)
            o.set("name", self.object_id + ":" + name)
            id_for_objects_with_no_name += 1

    @staticmethod
    def modify_property_helper(object_, element, kwargs):
        objects = object_.findall(".//" + element)
        for o in objects:
            for k, v in kwargs.items():
                o.set(k, v)

    def rename_helper_other_tags(self, element_tag, property_tag, parent):
        """
        Renames a given property tag inside element tag
        :param element_tag:
        :param property_tag:
        :param parent:
        :return:
        """
        objects = parent.findall(".//" + element_tag)
        for o in objects:
            element = o.get(property_tag)
            if element:
                o.set(property_tag, self.object_id + ":" + element)

    @staticmethod
    def search_and_rename(element_tag, property_tag, parent, renamed_element_tag, renamed_property_tag,
                          renamed_parent):
        """
        Search for the element with a given property in the worldbody and rename the corresponding element in the parent.
        Example:
        <mujoco>
            <contact>
                <exclude body1="base" body2="joint_1"/>
            </contact>
        </mujoco>
        :param element_tag: direct child of parent, e.g. exclude
        :param property_tag: property inside the element, e.g. body1 or body2
        :param parent: direct children of "mujoco" in the xml file, e.g. "contact", "asset", "worldbody", etc.
        :return:
        """
        renamed_objects = renamed_parent.findall(".//" + renamed_element_tag)

        unrenamed_objects = parent.findall(".//" + element_tag)  # need to rename all of these elements

        for uo in unrenamed_objects:
            unrenamed_element = uo.get(property_tag)
            # search for unrenamed_element in renamed_objects
            for o in renamed_objects:
                renamed_element = o.get(renamed_property_tag)
                if unrenamed_element in renamed_element:
                    # bingo
                    uo.set(property_tag, renamed_element)
                    """print("renamed {unrenamed_element} in {parent}.{element_tag} to {renamed_element}".format(
                        unrenamed_element=unrenamed_element,
                        parent=parent,
                        element_tag=element_tag,
                        renamed_element=renamed_element))"""
                    break

    def remove_contact(self, parent_body):
        """
        <contact>
        <exclude body1="link_0" body2="link_1"></exclude>
        </contact>
        :return:
        """
        child_bodies = parent_body.findall(".//body")
        if child_bodies is None:
            return

        body1 = parent_body

        for body2 in child_bodies:
            if body1.get("name") and body2.get("name"):
                exclude_contact = mjcf_utils.exclude_contact(body1.get("name"), body2.get("name"))
                self.contact.append(exclude_contact)
            self.remove_contact(body2)

    @property
    def joints(self):
        """
        :return:  joint names excluding the gripper or 7 (position, quaternion)
        """
        joints = self.worldbody.findall(".//joint")
        self.joint_names = [joint_name.get("name") or "" for joint_name in joints]
        return self.joint_names

    @property
    def joints_elements(self):
        """
        :return:  joint names excluding the gripper or 7 (position, quaternion)
        """
        return self.worldbody.findall(".//joint")

    @property
    def dof(self):
        """

        :return: number of joints excluding the gripper or 7 (position, quaternion)
        """
        if self._dof is None:
            self._dof = len(self.joints or [])
        return self._dof

    @dof.setter
    def dof(self, dof):
        """

        :return: number of joints excluding the gripper or 7 (position, quaternion)
        """
        self._dof = dof

    @property
    def joint_limits(self):
        if self.joint_limits_ is None:
            joints = self.worldbody.findall(".//joint")
            temp = []
            for joint in joints:
                if joint.get("range") is not None:
                    temp += [mjcf_utils.string_to_array(joint.get("range"))]
                else:
                    temp += [[-3.14, 3.14]]
            self.joint_limits_ = np.array(temp)
            # print(self.joint_limits_)
        return self.joint_limits_

    @staticmethod
    def get_unique_name(object_name):
        """
        Return a unique name for a given desired name. First object always have the exact assigned name so that we dont
        change the name if there is only one robot
        :param object_name: desired object name
        :return: object_name if it is unique, else object_name_{}.format(id)
        """
        global object_ids_dict
        existing_object_id = ObjectBase.object_ids_dict.get(object_name)
        if existing_object_id is None:
            unique_object_name = object_name
            existing_object_id = 1
        else:
            existing_object_id += 1
            unique_object_name = object_name + "_" + str(existing_object_id)
        ObjectBase.object_ids_dict[object_name] = existing_object_id
        return unique_object_name

    @property
    def grip_pos_site_name(self):
        grip_site = get_site(self.worldbody, "grip:grip")
        if grip_site is None:
            grip_site = get_site(self.worldbody, "ee_link")
        return grip_site.get("name")

    @property
    def grip_pos_body_name(self):
        grip_body = get_body(self.worldbody, "grip*")
        if grip_body is None:
            grip_body = get_body(self.worldbody, "ee_link")
        return grip_body.get("name")

    def add_model(self, model, body_name=None) -> bool:
        """
        Mounts model to self. To mount a gripper, make sure to use add_gripper!

        Args:
            Model (MujocoXml instance): gripper MJCF model
        :return model added successfully
        """
        if body_name:
            body_subtree = get_body(self.worldbody, body_name)
        else:
            body_subtree = self.base_body
        for body in model.worldbody:
            body_subtree.append(body)
        model.merged_already = True
        self.merge(model, merge_body=False)
        return True

    @property
    def all_bodies_names(self):
        if self.all_bodies_names_ is None:
            self.all_bodies_names_ = [body.get("name") for body in self.worldbody.findall(".//body")]
        return self.all_bodies_names_

    @property
    def all_geoms_names(self):
        if self.all_geoms_names_ is None:
            self.all_geoms_names_ = [geom.get("name") for geom in self.worldbody.findall(".//geom")]
        return self.all_geoms_names_

    def reduced_dict(self):
        self_dict = self.__dict__.copy()
        self_dict["type"] = self_dict["name"]
        remove_keys = ['tree', 'root', "name",
                       'worldbody', 'actuator', 'sensor', 'asset', 'equality', 'contact', 'default', 'tendon', 'visual',
                       'option',
                       'all_bodies_names_', 'all_geoms_names_', 'joint_names',
                       'joint_limits_', '_dof', "file", "folder",
                       'n_joints', 'control', 'control_using_mocap_not_joints', 'mocap_body']
        for k in remove_keys:
            self_dict.pop(k)
        for k, v in self_dict.items():
            if isinstance(v, np.ndarray):
                self_dict[k] = v.tolist()
        return self_dict

    def to_xml(self):
        self_dict = self.reduced_dict()
        from dicttoxml import dicttoxml
        from xml.dom.minidom import parseString
        xml = dicttoxml(self_dict, attr_type=False, custom_root=self.__class__.__name__)  # set root node
        dom = parseString(xml)
        return dom.toprettyxml()

    def set_base_xpos(self, pos):
        """Places the object on position @pos."""
        # node = self.worldbody.find("./body")  # TODO possible bug taking first body
        base_body = self.base_body
        if base_body is not None and not (isinstance(base_body, list) and len(base_body) == 0):
            nodes = base_body
        else:
            nodes = self.base_geoms
        nodes = [nodes] if not isinstance(nodes, Iterable) else nodes
        for node in nodes:
            node.set("pos", array_to_string(pos))

    @property
    def base_geoms(self):
        base_geoms = self.worldbody.findall(".//geom")
        return base_geoms

    def get_base_xpos(self):
        """Places the object on position @pos."""  # todo rename xpos to pos instead, since it is relative to parent body
        # node = self.worldbody.find("./body")  # TODO possible bug taking first body
        nodes = self.base_body or self.base_geoms
        nodes = [nodes] if not isinstance(nodes, Iterable) else nodes
        pos = [array_to_string(node.get("pos", "0. 0. 0.")) for node in nodes]
        return pos

    def get_base_xeuler(self, in_degrees=True):
        """Places the object on position @pos."""  # todo rename xeuler to euler instead, since it is relative to parent body
        # node = self.worldbody.find("./body")  # TODO possible bug taking first body
        nodes = self.base_body or self.base_geoms
        nodes = [nodes] if not isinstance(nodes, Iterable) else nodes
        euler = [array_to_string(node.get("euler", "0. 0. 0.")) for node in nodes]
        if in_degrees:
            euler = np.array(euler) * 180. / np.pi
        return euler

    def get_base_element_id_and_fn_from_sim(self, sim):
        """

        Args:
            sim:

        Returns:
            base_element_id: element id
            base_element_fn_src: sim.data or sim.model if body or geom respectively

        """
        base_element = self.base_body
        base_element_fn_src = sim.data
        base_element_id_fn = sim.model.body_name2id
        if base_element is None or (isinstance(base_element, list) and len(base_element) == 0):
            base_element = self.base_geoms[0]  # take only first geom #todo extend for all geoms
            base_element_fn_src = sim.model
            base_element_id_fn = sim.model.geom_name2id
        base_element_id = base_element_id_fn(base_element.get("name"))
        return base_element_id, base_element_fn_src

    def get_base_xpos_from_sim(self, sim):
        base_element_id, base_element_fn_src = self.get_base_element_id_and_fn_from_sim(sim)
        if base_element_id:
            if base_element_fn_src == sim.data:
                pos = base_element_fn_src.body_xpos[base_element_id]
            else:
                pos = base_element_fn_src.geom_pos[base_element_id]
            return pos

    def get_base_xeuler_from_sim(self, sim, in_degrees=True):
        base_element_id, base_element_fn_src = self.get_base_element_id_and_fn_from_sim(sim)
        if base_element_id:
            if base_element_fn_src == sim.data:
                quat = base_element_fn_src.body_xquat[base_element_id]
            else:
                quat = base_element_fn_src.geom_quat[base_element_id]
            euler = quat2euler(quat)
            if in_degrees:
                euler = euler * 180. / np.pi
            return euler

    def set_euler(self, euler):
        """sets euler position"""  # TODO possible bug taking first body
        # node = self.worldbody.find("./body")
        base_body = self.base_body
        if base_body is not None and not (isinstance(base_body, list) and len(base_body) == 0):
            nodes = base_body
        else:
            nodes = self.base_geoms
        nodes = [nodes] if not isinstance(nodes, Iterable) else nodes
        for node in nodes:
            node.set("euler", array_to_string(euler))

    def set_base_size(self, size):
        """Places the object on position @pos."""
        try:
            body = self.worldbody.find("./body/body").find("./body")
            node = body.find("./geom")
            node.set("size", array_to_string(size))
        except:
            print(node)
            pass
