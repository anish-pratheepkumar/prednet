import io
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
from shutil import copyfile

import lxml.etree as etree
import numpy as np

from .mjcf_utils import array_to_string
from ..utils import XMLError
from ..utils.transform_utils import quat_to_euler


class MujocoXML(object):
    """
    Base class of Mujoco xml file
    Wraps around ElementTree and provides additional functionality for merging different models.
    Specially, we keep track of <worldbody/>, <actuator/> and <asset/>
    """

    def __init__(self, fname):
        """
        Loads a mujoco xml from file.

        Args:
            fname (str): path to the MJCF xml file.
        """
        self.file = fname
        self.folder = os.path.dirname(fname)
        self.tree = None
        self.root = None
        self.name = None
        self.worldbody = None
        self.actuator = None
        self.sensor = None
        self.asset = None
        self.equality = None
        self.contact = None
        self.default = None
        self.tendon = None
        self.visual = None
        self.option = None

        self.initialize_base()

        # self.tree = ET.parse(fname)
        # self.root = self.tree.getroot()
        # self.name = self.root.get("model")
        # self.worldbody = self.create_default_element("worldbody")
        # self.actuator = self.create_default_element("actuator")
        # self.sensor = self.create_default_element("sensor")
        # self.asset = self.create_default_element("asset")
        # self.equality = self.create_default_element("equality")
        # self.contact = self.create_default_element("contact")
        # self.default = self.create_default_element("default")
        # self.tendon = self.create_default_element("tendon")
        # self.visual = self.create_default_element("visual")
        # self.option = self.create_default_element("option")
        # self.resolve_asset_dependency()

    def initialize_base(self):
        self.tree = ET.parse(self.file)
        self.root = self.tree.getroot()
        self.name = self.root.get("model")
        self.worldbody = self.create_default_element("worldbody")
        self.actuator = self.create_default_element("actuator")
        self.sensor = self.create_default_element("sensor")
        self.asset = self.create_default_element("asset")
        self.equality = self.create_default_element("equality")
        self.contact = self.create_default_element("contact")
        self.default = self.create_default_element("default")
        self.tendon = self.create_default_element("tendon")
        self.visual = self.create_default_element("visual")
        self.option = self.create_default_element("option")
        self.resolve_asset_dependency()

    def set_fname(self, fname):
        self.save_model(fname)
        self.file = fname
        self.folder = os.path.dirname(fname)
        self.asset = self.create_default_element("asset")
        self.resolve_asset_dependency()

    def resolve_asset_dependency(self):
        """
        Converts every file dependency into absolute path so when we merge we don't break things.
        """

        for node in self.asset.findall("./*[@file]"):
            file = node.get("file")
            abs_path = os.path.abspath(self.folder)
            abs_path = os.path.join(abs_path, file)
            node.set("file", abs_path)

    def create_default_element(self, name):
        """
        Creates a <@name/> tag under root if there is none.
        """

        found = self.root.find(name)
        if found is not None:
            return found
        ele = ET.Element(name)
        self.root.append(ele)
        return ele

    def merge(self, other, robot_name_list=None, merge_body=True):

        """
        Default merge method.

        Args:
            other: another MujocoXML instance
                raises XML error if @other is not a MujocoXML instance.
                merges <worldbody/>, <actuator/> and <asset/> of @other into @self
            merge_body: True if merging child bodies of @other. Defaults to True.
        """
        if not isinstance(other, MujocoXML):
            raise XMLError("{} is not a MujocoXML instance.".format(type(other)))

        # Keep hashtable with class_types for body, geom, site, joint, name

        if merge_body:
            for body in other.worldbody:
                self.worldbody.append(body)
        self.merge_asset(other)
        for one_actuator in other.actuator:
            self.actuator.append(one_actuator)
        for one_sensor in other.sensor:
            self.sensor.append(one_sensor)
        for one_equality in other.equality:
            self.equality.append(one_equality)
        for one_contact in other.contact:
            self.contact.append(one_contact)
        for one_default in other.default:
            if not robot_name_list or robot_name_list.count(other.name[2:]) > 1:
                # if default element of robot is appended once donot append again
                # if creating empty arena skip default
                pass
            else:
                self.default.append(one_default)
        for one_tendon in other.tendon:
            self.tendon.append(one_tendon)
        for one_visual in other.visual:
            self.visual.append(one_visual)

        # Update self.options.attribute from other, other has higher priority than self.
        self.option.attrib.update(other.option.attrib)

    def unmerge(self, other):
        """
        removes merged body
        """
        if not isinstance(other, MujocoXML):
            raise XMLError("{} is not a MujocoXML instance.".format(type(other)))

        # Keep hashtable with class_types for body, geom, site, joint, name

        for body in other.worldbody:
            for body_2 in self.worldbody.iter("body"):
                if body in body_2:
                    body_2.remove(body)
                    break
        self.unmerge_asset(other)
        for one_actuator in other.actuator:
            self.actuator.remove(one_actuator)
        for one_sensor in other.sensor:
            self.sensor.remove(one_sensor)
        for one_equality in other.equality:
            self.equality.remove(one_equality)
        for one_contact in other.contact:
            self.contact.remove(one_contact)
        for one_default in other.default:
            self.default.remove(one_default)
        for one_tendon in other.tendon:
            self.tendon.remove(one_tendon)
        for one_visual in other.visual:
            self.visual.remove(one_visual)

    def get_model(self, mode="mujoco_py"):
        """
        Generates a MjModel instance from the current xml tree.

        Args:
            mode (str): Mode with which to interpret xml tree

        Returns:
            MjModel: generated model from xml

        Raises:
            ValueError: [Invalid mode]
        """

        available_modes = ["mujoco_py"]

        with io.StringIO() as string:
            try:
                string.write(ET.tostring(self.root, encoding="unicode"))
                if mode == "mujoco_py":
                    model = load_model_from_xml(string.getvalue())
                    return model
                raise ValueError(
                    "Unkown model mode: {}. Available options are: {}".format(
                        mode, ",".join(available_modes)
                    )
                )
            except Exception as e:
                # print(e)
                fname = 'EXCEPTION.xml'
                # directory = os.path.dirname(os.path.abspath(fname))
                # if not os.path.exists(directory): os.mkdir(directory)
                # with open(fname, "w") as f:
                #     f.write(string.getvalue())
                xml_str = string.getvalue()
                parser = etree.XMLParser(remove_blank_text=True, encoding="UTF-8")
                elem = etree.XML(xml_str, parser=parser)
                with open(fname, "wb") as f:
                    f.write(etree.tostring(elem, pretty_print=True))

                print("Exception loading xml file. Saved exception to {}".format(os.path.abspath(fname)))

    def get_xml(self):
        """
        Returns a string of the MJCF XML file.
        """
        with io.StringIO() as string:
            string.write(ET.tostring(self.root, encoding="unicode"))
            return string.getvalue()

    def save_model(self, fname, with_stl_files=False, sim_to_save_state=None, **_):
        """
        Saves the xml to file.

        Args:
            fname: output file location
            pretty: attempts!! to pretty print the output
        """
        directory = os.path.dirname(os.path.abspath(fname))
        if not os.path.exists(directory): os.makedirs(directory)
        if sim_to_save_state is not None:
            # save all body, geom, site: pos, orientation
            for element in self.worldbody.findall(".//"):
                tag = element.tag
                element_name = element.get("name", None)
                if element_name is not None:
                    elem_id = sim_to_save_state.model.__getattribute__("{}_name2id".format(tag))(element_name)
                    if tag == "geom" and element.get("type") == "mesh":
                        # ignore mesh geoms, since mesh geom pos are processed by Mujoco, so we dont know the pos set by the user
                        continue
                    else:
                        pass
                    if "joint" in tag:
                        element.set("ref", str(sim_to_save_state.data.__getattribute__("qpos")[elem_id]))
                    else:
                        pass
                    try:
                        if tag == "body" and element.get("mocap", None) is not None:
                            element.set("pos", array_to_string(
                                sim_to_save_state.data.__getattribute__("{}_xpos".format(tag))[elem_id], precision=5))
                            element.set("quat",
                                        array_to_string(
                                            sim_to_save_state.data.__getattribute__("{}_xquat".format(tag))[elem_id],
                                            precision=4))
                        else:
                            element.set("pos", array_to_string(
                                sim_to_save_state.model.__getattribute__("{}_pos".format(tag))[elem_id], precision=5))
                            element_euler = element.get("euler", None)
                            if element_euler:
                                element.set("euler",
                                            array_to_string(quat_to_euler(
                                                sim_to_save_state.model.__getattribute__("{}_quat".format(tag))[
                                                    elem_id], degrees=False),
                                                precision=4))
                            else:
                                element.set("quat",
                                            array_to_string(
                                                sim_to_save_state.model.__getattribute__("{}_quat".format(tag))[
                                                    elem_id],
                                                precision=4))
                        size = sim_to_save_state.model.__getattribute__("{}_size".format(tag))[elem_id]
                        if np.sum(size) <= 1e-4:
                            pass
                        else:
                            element.set("size", array_to_string(size, precision=6))
                    except AttributeError as e:
                        # print(e)
                        continue
                else:
                    continue
        else:
            pass
        xml_str = ET.tostring(self.root, encoding="unicode")

        if with_stl_files:
            # copy all stl files to a location
            export_asset_dir = os.path.join(directory, "assets")
            prefix_dir = os.path.abspath(self.folder)
            sub_asset_files = list(self.asset)
            _replace_dirs = []
            for sub_asset in sub_asset_files:
                sub_asset_path = sub_asset.get("file", None)
                if sub_asset_path is None:
                    continue
                else:
                    if prefix_dir in sub_asset_path:
                        sub_export_asset_dir_ = sub_asset_path.replace(prefix_dir, export_asset_dir)
                    else:
                        sub_asset_path_temp = sub_asset_path if sub_asset_path[0] != "/" else sub_asset_path[1:]
                        _replace_dirs.append((sub_asset_path, sub_asset_path_temp))
                        sub_export_asset_dir_ = os.path.join(export_asset_dir, sub_asset_path_temp)
                    os.makedirs(os.path.dirname(sub_export_asset_dir_), exist_ok=True)
                    try:
                        copyfile(sub_asset_path, sub_export_asset_dir_)
                    except:
                        continue

            xml_str = xml_str.replace(prefix_dir + "/", "./assets/")
            for _replace_dir_name, _replace_dir_name_by in set(_replace_dirs):
                xml_str = xml_str.replace(_replace_dir_name, os.path.join("./assets", _replace_dir_name_by))

        parser = etree.XMLParser(remove_blank_text=True, encoding="UTF-8")
        elem = etree.XML(xml_str, parser=parser)
        with open(fname, "wb") as f:
            f.write(etree.tostring(elem, pretty_print=True))

        print("saved model to {}".format(fname))

    def get_model_as_string(self, pretty=True):
        """
        return the xml

        Args:
            fname: output file location
            pretty: attempts!! to pretty print the output
        """
        xml_str = ET.tostring(self.root, encoding="unicode")
        if pretty:
            # TODO: get a better pretty print library
            parsed_xml = xml.dom.minidom.parseString(xml_str)
            xml_str = parsed_xml.toprettyxml(newl="")
        # print('output xml: \current_timestep',xml_str)
        return xml_str

    def merge_asset(self, other):
        """
        Useful for merging other files in a custom logic.
        """
        for asset in other.asset:
            asset_name = asset.get("name")
            asset_type = asset.tag
            # Avoids duplication
            pattern = "./{}[@name='{}']".format(asset_type, asset_name)
            if self.asset.find(pattern) is None:
                self.asset.append(asset)

    def unmerge_asset(self, other):
        """
        Useful for unmerging other files in a custom logic.
        """
        for asset in other.asset:
            asset_name = asset.get("name")
            asset_type = asset.tag
            # Avoids duplication
            pattern = "./{}[@name='{}']".format(asset_type, asset_name)
            if self.asset.find(pattern) is not None:
                self.asset.remove(asset)

    # ---------------------------------------------
    #   Added the following from the latest robosuit
    # ---------------------------------------------
    def get_element_names(self, root, element_type):
        """
        Searches recursively through the @root and returns a list of names of the specified @element_type

        Args:
            root (ET.Element): Root of the xml element tree to start recursively searching through
                (e.g.: `self.worldbody`)
            element_type (str): Name of element to return names of. (e.g.: "site", "geom", etc.)

        Returns:
            list: names that correspond to the specified @element_type
        """
        names = []
        for child in root:
            if child.tag == element_type:
                names.append(child.get("name"))
            names += self.get_element_names(child, element_type)
        return names

    def add_prefix(self,
                   prefix,
                   tags=("body", "joint", "sensor", "site", "geom", "camera", "actuator", "tendon", "asset", "mesh",
                         "texture", "material")):
        """
        Utility method to add prefix to all body names to prevent name clashes
        TODO use this instead of rename functions in MujocoXMLObject
        Args:
            prefix (str): Prefix to be appended to all requested elements in this XML
            tags (list or tuple): Tags to be searched in the XML. All elements with specified tags will have "prefix"
                prepended to it
        """
        # Define tags as a set
        tags = set(tags)

        # Define equalities set to pass at the end
        equalities = set(tags)

        # Add joints to equalities if necessary
        if "joint" in tags:
            equalities = equalities.union(["joint1", "joint2"])

        # Handle actuator elements
        if "actuator" in tags:
            tags.discard("actuator")
            for actuator in self.actuator:
                self._add_prefix_recursively(actuator, tags, prefix)

        # Handle sensor elements
        if "sensor" in tags:
            tags.discard("sensor")
            for sensor in self.sensor:
                self._add_prefix_recursively(sensor, tags, prefix)

        # Handle tendon elements
        if "tendon" in tags:
            tags.discard("tendon")
            for tendon in self.tendon:
                self._add_prefix_recursively(tendon, tags.union(["fixed"]), prefix)
            # Also take care of any tendons in equality constraints
            equalities = equalities.union(["tendon1", "tendon2"])

        # Handle asset elements
        if "asset" in tags:
            tags.discard("asset")
            for asset in self.asset:
                if asset.tag in tags:
                    self._add_prefix_recursively(asset, tags, prefix)

        # Handle contacts and equality names for body elements
        if "body" in tags:
            for contact in self.contact:
                if "body1" in contact.attrib:
                    contact.set("body1", prefix + contact.attrib["body1"])
                if "body2" in contact.attrib:
                    contact.set("body2", prefix + contact.attrib["body2"])
            # Also take care of any bodies in equality constraints
            equalities = equalities.union(["body1", "body2"])

        # Handle all equality elements
        for equality in self.equality:
            self._add_prefix_recursively(equality, equalities, prefix)

        # Handle all remaining bodies in the element tree
        for body in self.worldbody:
            if body.tag in tags:
                self._add_prefix_recursively(body, tags, prefix)

    def _add_prefix_recursively(self, root, tags, prefix):
        """
        Iteratively searches through all children nodes in "root" element to append "prefix" to any named subelements
        with a tag in "tags"

        Args:
            root (ET.Element): Root of the xml element tree to start recursively searching through
                (e.g.: `self.worldbody`)
            tags (list or tuple): Tags to be searched in the XML. All elements with specified tags will have "prefix"
                prepended to it
            prefix (str): Prefix to be appended to all requested elements in this XML
        """
        # First re-name this element
        if "name" in root.attrib:
            root.set("name", prefix + root.attrib["name"])

        # Then loop through all tags and rename any appropriately
        for tag in tags:
            if tag in root.attrib:
                root.set(tag, prefix + root.attrib[tag])

        # Recursively go through child elements
        for child in root:
            if child.tag in tags:
                self._add_prefix_recursively(child, tags, prefix)


def load_model_from_xml(xml_str):
    """
    Loads and returns a PyMjModel model from a string containing XML markup.
    Saves the XML string used to create the returned model in `model.xml`.
    """
    from mujoco_py import load_model_from_path
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as fp:
        fp.write(xml_str.encode())
        fp.flush()
    model = load_model_from_path(fp.name)
    os.remove(fp.name)
    return model
